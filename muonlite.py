import math
import torch
import warnings
from typing import Callable, Iterable, Tuple
from loguru import logger
from itertools import repeat
import torch.nn.functional as F
import os
import re
__all__ = ["Muonlite"]


def beta_scheduler(t: int, warmup: int, beta_final: float, beta_start: float, T_beta: int) -> float:

    if T_beta>0:
        if t >= T_beta:
            exponent = beta_final
        elif t <= warmup:
            exponent = beta_start  

        else:
            exponent = beta_start-(beta_start-beta_final)*(t-warmup)/(T_beta-warmup)
    
    else:
        exponent = beta_final
    return exponent



def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr,min_ratio=0.1):
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        step (int): Current step (0-based).
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total training steps.
        max_lr (float): Maximum learning rate.
    
    Returns:
        float: Learning rate for the current step.
    """
    # 1. Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2. Cosine decay phase
    # Compute progress after warmup (0 to 1)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # Cosine decay to 0.1 * max_lr
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    decayed = (min_ratio + (1-min_ratio) * cosine_decay)  # Scales between 0.1 and 1.0
    return max_lr * decayed



coeffs_list = [
(8.28721201814563 , -23.595886519098837 , 17.300387312530933) ,
(4.107059111542203 , -2.9478499167379106 , 0.5448431082926601) ,
(3.9486908534822946 , -2.908902115962949 , 0.5518191394370137) ,
(3.3184196573706015 , -2.488488024314874 , 0.51004894012372) ,
(2.300652019954817 , -1.6689039845747493 , 0.4188073119525673) ,
(1.891301407787398 , -1.2679958271945868 , 0.37680408948524835) ,
(1.8750014808534479 , -1.2500016453999487 , 0.3750001645474248) ,
(1.875 , -1.25 , 0.375) , # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial )
coeffs_list = [( a / 1.01 , b / 1.01**3 , c / 1.01**5) for (a , b , c ) in coeffs_list [: -1]] + [ coeffs_list [ -1]]


class OptimizedNewtonSchulz:
    def __init__(self):
        self.shape_cache = {}
        self.stats = {"cache_hits": 0, "cache_misses": 0}
    
    def _get_compiled_function(self, shape_key: tuple):
        
        @torch.compile(
            dynamic=False,  
            fullgraph=True,  
            backend="inductor",
        )
        def compiled_func(G: torch.Tensor, steps: int) -> torch.Tensor:
            assert G.ndim >= 2
            X = G
            if G.size(-2) > G.size(-1):
                X = X.mT
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-8)
            
            hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
            for a, b, c in hs:
                A = X @ X.mT
                B = b * A + c * A @ A
                X = a * X + B @ X
            
            if G.size(-2) > G.size(-1):
                X = X.mT
            return X
        
        return compiled_func
    
    def __call__(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        shape_key = tuple(G.shape)
        
        
         
        if shape_key not in self.shape_cache:
            self.stats["cache_misses"] += 1
            print(f"Compiling new shape: {shape_key}, Cache size: {len(self.shape_cache) + 1}")
            self.shape_cache[shape_key] = self._get_compiled_function(shape_key)
        else:
            self.stats["cache_hits"] += 1
        
        return self.shape_cache[shape_key](G, steps)


zeropower_via_newtonschulz5 = OptimizedNewtonSchulz()




@torch.no_grad()
def mproj(m, msign_m, steps):
    """top subspace projection: keep eigenval>1 subspace
    """
    return (msign_m + zeropower_via_newtonschulz5(m - msign_m, steps) ) / 2


@torch.no_grad()
def rank_v(tensor, top_ratio=1.0,lower_ratio=0.5):
    """
    Perform piecewise linear scaling on a tensor:
      Elements smaller than lower_threshold are set to 0
      Elements greater than upper_threshold are set to 1.0
      Elements in [lower_threshold, upper_threshold] are linearly interpolated to [0, 1]

    Parameters:
        tensor: input tensor
        lower_threshold: lower bound  
        upper_threshold: upper bound  

    Returns:
        The processed tensor, which approximates a projection onto a sharp subspace.
    """

    lower_threshold= lower_ratio*torch.mean(tensor)
    upper_threshold= top_ratio*torch.mean(tensor)
    
    range_size = upper_threshold - lower_threshold+1e-12
    
    
    result = torch.zeros_like(tensor)
    
    
    mask_high = tensor >= upper_threshold
    result[mask_high] = 1.0
    
    
    mask_middle = (tensor >= lower_threshold) & (tensor < upper_threshold)
    
     
    if mask_middle.any():
        
        middle_values = tensor[mask_middle]
        
        normalized_values = (middle_values - lower_threshold) / range_size
        
        result[mask_middle] = normalized_values
    
    
    new_top_ratio=mask_high.sum()/tensor.numel()
    new_lower_ratio=torch.sum(tensor >= lower_threshold)/tensor.numel()
    return new_top_ratio, new_lower_ratio, result


class Muonlite(torch.optim.Optimizer):
    """Muonlite — Optimizer.

    The optimizer replicates the reference implementation shipped with
    Moonlight. 2-D weight matrices (except embeddings / lm_head) are updated via
    Muon, while all other parameters fall back to AdamW with decoupled weight
    decay.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.1,
        muon_params=None,
        subspace_ratio_dict=None,
        lr_dict=None,
        muon_theta: float = 0.95,
         
        beta1: float = 0.0,
        beta2: float = 0.0,
         
        ns_steps: int = 6,
        adamw_params=None,
        adamw_eps: float = 1e-8,
        adamw_b2: float = 0.95,
        adamw_theta: float = 0.9,
        
        T_flat_warmup = None ,  
         
        max_iter: int = 10000,

        warm_up_iter: int = 1000,
        min_lr_ratio: int = 0.1,
        lr_schedule = 'cosine',
        smooth_ratio: float = 0.1,


    ) -> None:
        defaults = dict(
            lr=lr,
            wd=wd,
            muon_theta=muon_theta,

            beta1=beta1,
            beta2=beta2,
            ns_steps=ns_steps,
            adamw_eps=adamw_eps,
            
            max_iter=max_iter,

            warm_up_iter=warm_up_iter,
            min_lr_ratio=min_lr_ratio,

           

        )
        
        
        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = [p for _, p in muon_params + adamw_params]  
        super().__init__(params, defaults)

        self.qk_modules_list = ["q_proj.weight", "k_proj.weight"]
        self.vo_modules_list = ["v_proj.weight", "o_proj"]
        self.ffn_modules_list = ["gate_proj", "down_proj", "up_proj"]
        self.router_modules_list = ["mlp.gate.weight"]
        self.emb_modules_list = ["embed"]
        self.out_modules_list = ["head"]
        self.norm_modules_list= ["norm"]
        
        self.warm_up_iter = warm_up_iter
        self.max_iter = max_iter
        if T_flat_warmup is None:
            if lr_schedule=='cosine':
                self.T_f=0.5
            else:
                self.T_f=2*self.warm_up_iter/self.max_iter     
        else:
            
            self.T_f=T_flat_warmup

              
        
        
        
        for name,p in muon_params:
            
            assert p.ndim == 2, "muon only supports 2-D parameters"
            self.state[p]["use_muon"] = 2
        for name,p in adamw_params:
            self.state[p]["use_muon"] = 0

            
        
        for param_name, p in muon_params + adamw_params:
            
            param_name=param_name.lower()
            self.state[p]["name"]=param_name
            if any(name in param_name for name in self.qk_modules_list):
            
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['qk']
                self.state[p]["lr_ratio"]=lr_dict['qk']
                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=1
                else:
                    self.state[p]["flat_warmup"]=0
                    
                if subspace_ratio_dict['qk']>1e-6:
                    self.state[p]["use_muon"] = 1

                

            elif any(name in param_name for name in self.vo_modules_list):
            
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['vo']
                self.state[p]["lr_ratio"]=lr_dict['vo']
                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=1
                else:
                    self.state[p]["flat_warmup"]=0
                    
                if subspace_ratio_dict['vo']>1e-6:
                    self.state[p]["use_muon"] = 1

                

            elif any(name in param_name for name in self.emb_modules_list):
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['emb']
                self.state[p]["lr_ratio"]=lr_dict['emb']

                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=2
                else:
                    self.state[p]["flat_warmup"]=0
                
                
                 

            elif any(name in param_name for name in self.out_modules_list):
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['out']

                self.state[p]["lr_ratio"]=lr_dict['out']
                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=2
                else:
                    self.state[p]["flat_warmup"]=0
                
                 

            elif any(name in param_name for name in self.norm_modules_list):
                
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['norm']
                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=2
                else:
                    self.state[p]["flat_warmup"]=0

                
                self.state[p]["lr_ratio"]=lr_dict['norm']
                
                
                 
                
            elif any(name in param_name for name in self.ffn_modules_list):
            
                self.state[p]["subspace_ratio"]=subspace_ratio_dict['ffn']
                self.state[p]["lr_ratio"]=lr_dict['ffn']
                if lr_schedule=='cosine':
                    
                    self.state[p]["flat_warmup"]=1
                else:
                    self.state[p]["flat_warmup"]=0
                if subspace_ratio_dict['ffn']>1e-6:
                    self.state[p]["use_muon"] = 1
                                               
                    
            else:  
                self.state[p]["subspace_ratio"]=1.0 
                self.state[p]["lr_ratio"]=1.0
                self.state[p]["flat_warmup"]=0
                

                

        
        self.min_lr_ratio=min_lr_ratio
        
        
        
        
        self.iter=0

        self.adamw_b2=adamw_b2
        self.adamw_theta=adamw_theta
        
        self.smooth_ratio=smooth_ratio
        

    
    
    def get_lr_ratio(self,lr_ratio,flat_mode):
            
        if flat_mode==0 or flat_mode is None:
            lr_ratio_t=lr_ratio

        elif flat_mode==2:
            lr_f=cosine_lr_schedule(self.iter, self.warm_up_iter, self.max_iter, lr_ratio,self.min_lr_ratio/lr_ratio)
            lr_s=cosine_lr_schedule(self.iter, self.warm_up_iter, self.max_iter, 1.0,self.min_lr_ratio)+1e-9
            lr_ratio_t=lr_f/lr_s   
        
        elif flat_mode==1:
            if self.iter<=self.warm_up_iter:
                lr_ratio_t=1.0
            else:
                lr_f=cosine_lr_schedule(self.iter, self.warm_up_iter, self.max_iter, lr_ratio,self.min_lr_ratio/lr_ratio)
                lr_s=cosine_lr_schedule(self.iter, self.warm_up_iter, self.max_iter, 1.0,self.min_lr_ratio)+1e-9
                T_f=self.T_f
                lr_ratio_t=lr_f/lr_s
                lr_ratio_t=min(lr_ratio_t,1+(self.iter-self.warm_up_iter)/(self.max_iter-self.warm_up_iter)*(lr_ratio-1)/T_f)                        
         
        return lr_ratio_t      
        
            
    @torch.no_grad()
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            lr, wd = group["lr"], group["wd"]
            muon_theta,beta1,beta2, ns_steps = group["muon_theta"],group["beta1"],group["beta2"], group["ns_steps"]
            
            
            T_beta=int(self.T_f*(self.max_iter-self.warm_up_iter))+self.warm_up_iter
            beta2=beta_scheduler(t=self.iter, warmup=self.warm_up_iter, beta_final=group["beta2"], beta_start=group["beta1"], T_beta=T_beta)
                                        
            # ------- Muon params (2-D) -------
            for p in [q for q in group["params"] if self.state[q]["use_muon"]==1]:
                g = p.grad
                state = self.state[p]                
                if g is None:
                    continue                
                                                    
                m, n = g.size(0),g.size(1)
                if "step" not in state:
                    state.update(step=0)
                    state.update(subspace_threshold_ratio=1.0/math.sqrt(min(m,n)))
                    state["momentum"] = 0.0*g
                
                                      

                k=int(state["subspace_ratio"]*min(m,n))+1
                k=min(k,min(m,n)-1)                        
                    
                lr_times=self.get_lr_ratio(state['lr_ratio'],state["flat_warmup"])
                
               

                state["momentum"]=state["momentum"]*muon_theta+g*(1-muon_theta)
                M=state["momentum"]+g*(1-muon_theta)/muon_theta
                    
                
                
                m_ns=zeropower_via_newtonschulz5(M, ns_steps)
                
                
               
                thres_r=M.norm()*state['subspace_threshold_ratio']+1e-9
                
                
                
                state_P=mproj(M/thres_r, m_ns, ns_steps) 
             
                
                if state_P.norm() > math.sqrt(k): 
                    state['subspace_threshold_ratio']=state['subspace_threshold_ratio']*1.05
                else:
                    state['subspace_threshold_ratio']=state['subspace_threshold_ratio']*0.95
                    
                state['subspace_threshold_ratio']=min(1.0,state['subspace_threshold_ratio'])

                P_flat_smooth=torch.eye(n, dtype=state_P.dtype, device=state_P.device)-state_P.t()@state_P

                hessian_damping=zeropower_via_newtonschulz5(g, ns_steps)
                hessian_damping_flat=hessian_damping@P_flat_smooth                    



                update=m_ns +beta1*hessian_damping+(beta2-beta1)*hessian_damping_flat
                update=update+(lr_times-1)*update@P_flat_smooth

                flat_data_wd=p.data@P_flat_smooth #extra weight decay in the flat directions
               
                
                if state["step"]==0:
                    name=state["name"]
                    logger.info(f"{name},use muonlite with beta1={group['beta1']},beta2={group['beta2']},subspace={state['subspace_ratio']},lr_ratio={state['lr_ratio']}")    
                    

                
                state["step"] += 1

                p.data.mul_(1 - lr * wd)
                
                wd2=(lr_times-1)* wd                               
                p.data.add_(flat_data_wd, alpha=- lr * wd2)

                
                p.data.add_(update, alpha=-0.2*lr*math.sqrt(max(m, n)) )
 
            # ------- Vanilla Muon params-------
            for p in [q for q in group["params"] if self.state[q]["use_muon"]==2]:
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state.update(step=0)
                
                if g is None:
                    continue
                
                
                
                m, n = g.size(0),g.size(1)
                
                if "momentum" not in state:
                    state["momentum"] = 0.0*g
                 
                
                state["momentum"]=state["momentum"]*muon_theta+g*(1-muon_theta)
                
                
                M=state["momentum"]+g*(1-muon_theta)/muon_theta
                
                u=zeropower_via_newtonschulz5(M , ns_steps)
                
                if state["step"]==0:
                    name=state["name"]
                    logger.info(f"{name},use_vanilla_muon")    
                
  

                state["step"] += 1
                
                p.data.mul_(1 - lr * wd)
                
                    
                p.data.add_(u, alpha=-0.2*lr*math.sqrt(max(m, n)))
                
            # ------- AdamW  params (emb out norm) -------
            
            eps = group["adamw_eps"]
            adam_b2=self.adamw_b2
            adam_theta=self.adamw_theta
            
            for p in [q for q in group["params"] if self.state[q]["use_muon"]==0]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state.update(step=0, moment1=0.0*g, moment2=0.0*g,P=None)
                    state['upper_topk_ratio']=1.0
                    state['lower_topk_ratio']=0.5
                    
                if state["step"]==0:
                    name=state["name"]
                    logger.info(f"{name},subspace={state['subspace_ratio']},lr_ratio={state['lr_ratio']}, adamw")
                    
                 
                    

                
                state["moment1"]=adam_theta*state["moment1"]+(1-adam_theta)*g
                state["moment2"]=state["moment2"]*adam_b2+(g**2)*(1 - adam_b2)
                
                
                lr_times=self.get_lr_ratio(state['lr_ratio'],state["flat_warmup"])


                smooth_ratio=min(1.0-state["subspace_ratio"],self.smooth_ratio)
                
                new_upper_topk_ratio,new_lower_topk_ratio,state_p=rank_v(state["moment2"],state['upper_topk_ratio'],state['lower_topk_ratio'])
                
                if new_lower_topk_ratio>smooth_ratio+state["subspace_ratio"]:
                    state['lower_topk_ratio']*=1.05
                else:
                    state['lower_topk_ratio']*=0.95
                if new_upper_topk_ratio>state["subspace_ratio"]:
                    state['upper_topk_ratio']*=1.05
                else:
                    state['upper_topk_ratio']*=0.95
                    
                state['lower_topk_ratio']=min(state['lower_topk_ratio'],state['upper_topk_ratio']*0.95)  
                
                if state["subspace_ratio"]<1e-5:
                    state_p=0.0*g
                elif state["subspace_ratio"]>1.0-1e-5:
                    state_p=0.0*g+1.0
                    
               
                
                update = state["moment1"]  / (state["moment2"].sqrt() + eps)
                bias_correction1 = 1 - adam_theta**(state["step"]+1)
                bias_correction2 = 1 - adam_b2**(state["step"]+1)
                scale = bias_correction1 / bias_correction2**0.5
                
                update=update+(lr_times-1)*(1-state_p)*update
                
                flat_data_wd= (1-state_p)*p.data
                wd2=(lr_times-1)* wd
                p.data.mul_(1 - lr * wd)
                p.data.add_(flat_data_wd, alpha=- lr * wd2)  
                
                

                p.data.add_(update, alpha=-lr/ scale)
                
                state["step"] += 1
            
            
        self.iter+=1
        return loss
