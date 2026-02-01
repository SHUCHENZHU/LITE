
import sys
sys.modules['transformer_engine'] = None
import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed

import torch._dynamo
torch._dynamo.config.optimize_ddp = False

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

from transformers import DataCollatorWithFlattening

from muonlite import Muonlite

transformers.logging.set_verbosity_error()

# TensorBoard for visualization
# SwanLab for visualization
import swanlab

# Allow overriding the API key via environment variable; fall back to the default used in swanlab.py
SWANLAB_API_KEY="E9QOahmrF2igWgV7P7S1x" #swanlab token
swanlab.login(api_key=SWANLAB_API_KEY)



def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_torch_compile", default=False, action="store_true")
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="2nd-order-optimizer")
    parser.add_argument("--tensorboard_dir", type=str, default="./runs", help="Directory for TensorBoard logs")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    
    parser.add_argument("--total_eval_token_number", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts","wsd"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    
    
    parser.add_argument("--subspace_ratio_emb", type=float, default=0.5)
    parser.add_argument("--subspace_ratio_norm", type=float, default=1.0)
    parser.add_argument("--subspace_ratio_out", type=float, default=0.5)
    parser.add_argument("--subspace_ratio_qk", type=float, default=0.5)
    parser.add_argument("--subspace_ratio_vo", type=float, default=0.5)
    parser.add_argument("--subspace_ratio_ffn", type=float, default=0.5)


    parser.add_argument("--lr_ratio_emb", type=float, default=1.0)
    parser.add_argument("--lr_ratio_out", type=float, default=1.0)
    parser.add_argument("--lr_ratio_qk", type=float, default=1.0)
    parser.add_argument("--lr_ratio_vo", type=float, default=1.0)
    parser.add_argument("--lr_ratio_ffn", type=float, default=1.0)
    parser.add_argument("--lr_ratio_norm", type=float, default=1.0)


    parser.add_argument("--adam_b2", type=float, default=0.99)

    parser.add_argument("--ns_steps", type=int, default=6)



    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=1000000000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--dataloader_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   

    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=1.0)

    parser.add_argument("--beta_shampoo", type=float,  default=0.99)
    parser.add_argument("--stable_ratio", type=float, default=0.8)

    parser.add_argument("--muon_theta", type=float, default=0.95)
    parser.add_argument("--adamw_theta", type=float, default=0.9)
    


    
    parser.add_argument("--smooth_ratio", type=float, default=0.1)
    
    
    
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    # FlashAttention-2 and sequence packing
    parser.add_argument("--use_flash_attention_2", default=False, action="store_true",
                       help="Enable FlashAttention-2 for efficient attention computation")
    parser.add_argument("--use_packing", default=False, action="store_true",
                       help="Enable sequence packing with DataCollatorWithFlattening")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, 
                   tokenizer=None, use_packing=False, max_length=1024,total_eval_token_number=None, single_gpu=False):
    _time = time.time()
    
    # Load Pile validation dataset
    data_file=args.data_file
    val_data = datasets.load_dataset(
        "json",
        data_files={"validation":f"{data_file}/val.jsonl.zst"},
        split="validation",
        streaming=True
    )
    val_data = val_data.shuffle(seed=42, buffer_size=10_000)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")
    
    if not single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    if total_eval_token_number is not None:
        target_eval_tokens = total_eval_token_number
    else:
        target_eval_tokens = 10_000_000
    logger.info(f"target_eval_tokens = {target_eval_tokens}")

    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1

    if use_packing and tokenizer is not None:
        logger.info("Using DataCollatorWithFlattening for validation")
        
        # Tokenize without padding for packing
        def tokenize_function_eval(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            return result
        
        val_data_mapped = val_data.map(
            tokenize_function_eval,
            batched=True,
            remove_columns=["text", "meta"],
        )
        
        data_collator = DataCollatorWithFlattening()
        
        class TokenizedIterableDatasetEval(torch.utils.data.IterableDataset):
            def __init__(self, dataset, max_samples=None):
                self.dataset = dataset
                self.max_samples = max_samples
                
            def __iter__(self):
                count = 0
                for example in self.dataset:
                    if self.max_samples is not None and count >= self.max_samples:
                        break
                    yield example
                    count += 1
        
        # Estimate max samples based on target tokens (rough estimate)
        max_samples = target_eval_tokens // (max_length // 2)  # Conservative estimate
        iterable_dataset = TokenizedIterableDatasetEval(val_data_mapped, max_samples=max_samples)
        
        val_dataloader = torch.utils.data.DataLoader(
            iterable_dataset,
            batch_size=batch_size,
            num_workers=1,
            collate_fn=data_collator,
        )
        
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
        
        for batch in val_dataloader:
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            
            # With DataCollatorWithFlattening, labels are already in batch
            if "labels" not in batch:
                batch["labels"] = batch["input_ids"].clone()
            
            loss = model(**batch).loss
            total_loss += loss.detach()

            # Count actual tokens (excluding -100)
            evaluated_on_tokens += (batch["labels"] != -100).sum().item() * world_size
    else:
        # Standard evaluation with padding
        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "meta"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

        for batch in val_data_mapped.batch(batch_size=batch_size):
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            loss = model(**batch, labels=labels).loss
            total_loss += loss.detach()

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches
    
    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"



    # turn off logger
    if global_rank != 0: logger.remove()
    
    if args.eval_batch_size is None:
        args.eval_batch_size= args.batch_size 
    # Initialize SwanLab run (only on rank 0)
    swan_run = None
    if global_rank == 0:
        
        name=f"lr={args.lr},opt={args.optimizer}"
            
        swan_run = swanlab.init(
            project=args.project_name,
            name=name,
            settings=swanlab.Settings(api_key=SWANLAB_API_KEY),
        )
        logger.info(f"SwanLab run initialized with name: {name}")
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # Load Pile dataset
    DATASET_PATH = args.data_file
    
    logger.info(f"Loading Pile dataset from {DATASET_PATH}")
    data = datasets.load_dataset(
        "json",
        data_files=f"{DATASET_PATH}/train/*.jsonl.zst",
        split="train",
        streaming=True
    )
    seed_for_shuffle = 42 
    
    logger.info(f"Shuffling Pile dataset with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle, buffer_size=10_000)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    
    tokenizer = AutoTokenizer.from_pretrained("./llama2tokenizer", model_max_length=args.max_length)
   
    
    # Preprocess function for evaluation (uses padding for simplicity)
    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch
    
    if args.use_packing:
        logger.info("Using DataCollatorWithFlattening for sequence packing")
        
        # Tokenize dataset without padding - DataCollatorWithFlattening will handle packing
        def tokenize_function(examples):
            # Tokenize without padding, truncate to max_length
            # DataCollatorWithFlattening expects input_ids field
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_length,
                padding=False,  # No padding - collator will pack sequences
                add_special_tokens=True,
            )
            return result
        
        # Map tokenization over the dataset
        tokenized_data = data.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "meta"],
        )
        
        # Create DataCollatorWithFlattening for efficient sequence packing
        # Note: DataCollatorWithFlattening doesn't take any arguments
        data_collator = DataCollatorWithFlattening()
        
        # For streaming datasets, we need to convert to iterable format
        class TokenizedIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, dataset):
                self.dataset = dataset
                
            def __iter__(self):
                # Cycle indefinitely over the tokenized streaming dataset
                while True:
                    for example in self.dataset:
                        yield example
        
        iterable_dataset = TokenizedIterableDataset(tokenized_data)
        # Streaming datasets typically expose a single shard per worker process.
        # Use a single DataLoader worker to avoid "Too many dataloader workers" errors.
        
        
        
        if args.dataloader_workers is None:
            
            dataloader = torch.utils.data.DataLoader(
                iterable_dataset,
                batch_size=args.batch_size,
                num_workers=1,
                pin_memory=True,      
                collate_fn=data_collator,
            )
        
        else:
            
            dataloader = torch.utils.data.DataLoader(
                iterable_dataset,
                batch_size=args.batch_size,
                num_workers=args.dataloader_workers,
                pin_memory=True,   
                prefetch_factor=4,   
                collate_fn=data_collator,
            )
    else:
        logger.info("Using standard PreprocessedIterableDataset with padding")
        dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
        # Use a single DataLoader worker for HF streaming datasets to respect dataset.num_shards
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1)

    model_config = AutoConfig.from_pretrained(args.model_config)
    
    
    # Configure attention implementation
    attn_implementation = "flash_attention_2" if args.use_flash_attention_2 else "eager"
    if args.use_flash_attention_2:
        logger.info("Using FlashAttention-2 for efficient attention computation")
        model_config._attn_implementation = "flash_attention_2"
    
    if args.use_hf_model or args.use_packing:
        # Use HF model for FlashAttention-2 and packing support
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(
            model_config,
            attn_implementation=attn_implementation
        )
    else:
        # Use custom model (no FlashAttention-2 support)
        model = LlamaForCausalLM(model_config)






    if args.activation_checkpointing:
        #model.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.use_torch_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            logger.info(f"[{time.time():.2f}] Compiling model with torch.compile for H100 optimization")
            # mode="reduce-overhead" is suitable for training scenarios
            model = torch.compile(model)
            logger.info(f"[{time.time():.2f}] Model compilation completed")
        except Exception as e:
            logger.warning(f"[{time.time():.2f}] torch.compile failed, continuing without compilation: {e}")
    else:
        if not args.use_torch_compile:
            logger.info(f"[{time.time():.2f}] torch.compile disabled (use --use_torch_compile to enable)")




    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'pile',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "use_flash_attention_2": args.use_flash_attention_2,
        "use_packing": args.use_packing,
        "attn_implementation": attn_implementation,
    })

    if global_rank == 0:
        swanlab.config.update(run_config, allow_val_change=True)
        swanlab.log({"script_path": os.path.abspath(__file__)}, step=0)
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    
    if args.optimizer.lower() == "muonlite":
        # Separate parameters: use Muon for 2-D transformer trainable weights and fallback AdamW for the rest
        muon_params = [
            (name, p) 
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            (name, p) 
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        
        subspace_ratio_dict={'emb':args.subspace_ratio_emb,'out':args.subspace_ratio_out,'norm':args.subspace_ratio_norm,'qk':args.subspace_ratio_qk,'vo':args.subspace_ratio_vo,'ffn':args.subspace_ratio_ffn}
        lr_ratio_dict={'emb':args.lr_ratio_emb,'out':args.lr_ratio_out,'norm':args.lr_ratio_norm,'qk':args.lr_ratio_qk,'vo':args.lr_ratio_vo,'ffn':args.lr_ratio_ffn}
        optimizer = Muonlite(
            lr=args.lr,
            wd=args.weight_decay,
            subspace_ratio_dict=subspace_ratio_dict,
            lr_dict=lr_ratio_dict,
            muon_theta=args.muon_theta,
            beta1=args.beta1,
            beta2=args.beta2,           
            ns_steps=args.ns_steps,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_b2=args.adam_b2,
            adamw_theta=args.adamw_theta,
            adamw_eps= 1e-8,            
            lr_schedule=args.scheduler,            
            max_iter=args.num_training_steps,
            warm_up_iter=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            smooth_ratio=args.smooth_ratio,
        )     

    
        
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
   
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        stable_ratio=args.stable_ratio,
    )
    
        
        
    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    def _unwrap_model(model_obj):
        if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
            return model_obj.module
        return model_obj

    def save_training_checkpoint(save_path, optimizer_state, training_state):
        os.makedirs(save_path, exist_ok=True)
        model_to_save = _unwrap_model(model)
        torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        torch.save(optimizer_state, os.path.join(save_path, "optimizer.pt"))
        with open(os.path.join(save_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=4)

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    unf_precondition_acc=0
    grad_accumulation=args.gradient_accumulation
    
    for batch_idx, batch in enumerate(dataloader):

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break        
        global_step += 1
        local_step += 1
                            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Handle labels based on whether packing is enabled
        if args.use_packing:
            # With DataCollatorWithFlattening, labels are already in batch
            if "labels" not in batch:
                # If labels not present, create them from input_ids
                batch["labels"] = batch["input_ids"].clone()
            # Count actual tokens (excluding -100)
            tokens_seen += (batch["labels"] != -100).sum().item() * world_size
        else:
            # Standard approach: create labels from input_ids
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            batch["labels"] = labels
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
# -----------------------------------------------------------------------------------------

        loss = model(**batch).loss
        scaled_loss = loss /  grad_accumulation
        scaled_loss.backward()

        
            
        # with open("training_log_text.txt", "a") as f:
             
        #     f.write(str({k: v.shape for k, v in batch.items()}) + "\n")
        #     sample_idx = 0
        #     f.write("\n=== Tokenization Check ===\n")
        #     f.write(f"Input IDs: {batch['input_ids'][sample_idx].tolist()}\n")
        #     f.write(f"Labels: {labels[sample_idx].tolist()}\n")
        #     f.write(f"Decoded Input: {tokenizer.decode(batch['input_ids'][sample_idx])}\n")
        #     f.write(f"Pad Tokens Count: {(batch['input_ids'] == pad_idx).sum().item()}\n")
        if global_step % grad_accumulation != 0:
            continue



        # The below code is only executed during the update step
        
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        
        update_step += 1
        if global_rank == 0 and swan_run is not None:
            block_metrics = getattr(optimizer, "last_block_metrics", None)
            if block_metrics:
                for block_name, metrics in block_metrics.items():
                    if metrics["grad_norm"] == 0 and metrics["update_norm"] == 0:
                        continue
                    swanlab.log(
                        {
                            f"blocks/{block_name}/grad_norm": metrics["grad_norm"],
                            f"blocks/{block_name}/update_norm": metrics["update_norm"],
                        },
                        step=update_step,
                    )
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > grad_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "dtype": args.dtype,
            }

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            save_training_checkpoint(current_model_directory, optimizer_checkpoint, training_state_checkpoint)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.eval_batch_size,
                tokenizer=tokenizer, use_packing=args.use_packing, max_length=args.max_length,total_eval_token_number=args.total_eval_token_number, single_gpu=args.single_gpu
            )
            if global_rank == 0:
                swanlab.log(
                    {
                        "eval/loss": total_loss,
                        "eval/tokens": evaluated_on_tokens,
                    },
                    step=update_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_seen": tokens_seen,
                    "throughput/tokens_per_sec": tokens_in_update / update_time,
                    "throughput/examples_per_sec": args.total_batch_size / update_time,
                    "throughput/batches_per_sec": batches_in_update / update_time,
                },
                step=update_step,
            )
        update_time = time.time()

# -----------------------------------------------------------------------------------------
    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "dtype": args.dtype,
        }
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        save_training_checkpoint(current_model_directory, optimizer_checkpoint, training_state_checkpoint)





    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size,
        tokenizer=tokenizer, use_packing=args.use_packing, max_length=args.max_length, single_gpu=args.single_gpu
    )

    if global_rank == 0:
        swanlab.log(
            {
                "final_eval/loss": total_loss,
                "final_eval/tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")
        logger.info("SwanLab logging finished")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
