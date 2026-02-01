
export CUDA_VISIBLE_DEVICES=0,1,2,3


BS=${BS:-32}   # micro batch size. It has minimal impact when fixing TOTAL_BS.
EVAL_BS=${EVAL_BS:-32} # micro batch size for evalutaion. IMPORTANT: To ensure fairness, this value (EVAL_BS), along with the number of GPUs (such as 4) and the evaluation tokens (EVAL_TOKEN), must be set to the same values when comparing different algorithms. This guarantees that the allocated evaluation set remains consistent. A typical choice is the value of BS.
EVAL_TOKEN=${EVAL_TOKEN:-10000000} 

TOTAL_BS=${TOTAL_BS:-1024}
seq_length=${seq_length:-1024} 

TOTAL_ITERATION=${TOTAL_ITERATION:-10000} 
PROJECT_NAME="pile-pretrain-250m-muonlite-cos"

data_path='/huggingface/monopoly-pile-uncopyrighted'

LR=${LR:-0.002} 
WARM_UP=${WARM_UP:-1000}   # warmup iterations    
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}   #the value of lr_min/lr_max  in cosine/wsd lr schedules

muon_theta=${muon_theta:-0.95}  

BETA1=${BETA1:-0.0} #{-0.25,0.0}
BETA2=${BETA2:-0.0} #{0.5,1.0,2.0}

adamw_theta=${adamw_theta:-0.9} 

# LR_RATIO corresponds chi  in the paper
LR_RATIO_emb=${LR_RATIO_emb:-1.0}
LR_RATIO_out=${LR_RATIO_out:-1.0}
LR_RATIO_norm=${LR_RATIO_norm:-1.0}
LR_RATIO_qk=${LR_RATIO_qk:-1.0}
LR_RATIO_vo=${LR_RATIO_vo:-1.0}
LR_RATIO_ffn=${LR_RATIO_ffn:-1.0}

SUBSPACE_RATIO_emb=${SUBSPACE_RATIO_emb:-1.0}
SUBSPACE_RATIO_out=${SUBSPACE_RATIO_out:-1.0}
SUBSPACE_RATIO_norm=${SUBSPACE_RATIO_norm:-1.0}
SUBSPACE_RATIO_qk=${SUBSPACE_RATIO_qk:-0.0}
SUBSPACE_RATIO_voffn=${SUBSPACE_RATIO_voffn:-0.0}

NS_STEP=${NS_STEP:-6} #num_steps for Newton Schulz iterations
LRS=${LRS:-cosine}  #cosine/wsd
STABLE=${STABLE:-0.8} #only work when setting LRS=wsd 
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}  

echo "Running training with LR=$LR"
torchrun --standalone --nproc-per-node 4  pretrain_pile_llama2.py \
  --model_config configs/llama_500m.json \
  --grad_clipping 1.0 \
  --project_name "$PROJECT_NAME" \
  --lr $LR \
  --min_lr_ratio $MIN_LR_RATIO \
  --data_file $data_path \
  --batch_size $BS \
  --eval_batch_size $EVAL_BS \
  --total_eval_token_number $EVAL_TOKEN \
  --total_batch_size $TOTAL_BS \
  --max_length $seq_length \
  --num_training_steps $TOTAL_ITERATION \
  --subspace_ratio_emb $SUBSPACE_RATIO_emb \
  --subspace_ratio_out $SUBSPACE_RATIO_out \
  --subspace_ratio_norm $SUBSPACE_RATIO_norm \
  --subspace_ratio_qk $SUBSPACE_RATIO_qk \
  --subspace_ratio_vo $SUBSPACE_RATIO_voffn \
  --subspace_ratio_ffn $SUBSPACE_RATIO_voffn \
  --lr_ratio_emb $LR_RATIO_emb \
  --lr_ratio_out $LR_RATIO_out \
  --lr_ratio_qk $LR_RATIO_qk \
  --lr_ratio_vo $LR_RATIO_vo \
  --lr_ratio_ffn $LR_RATIO_ffn \
  --lr_ratio_norm $LR_RATIO_norm \
  --muon_theta $muon_theta \
  --adamw_theta $adamw_theta \
  --warmup_steps $WARM_UP \
  --weight_decay $WEIGHT_DECAY \
  --ns_steps $NS_STEP \
  --dtype bfloat16 \
  --eval_every 500 \
  --save_every 20000 \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --stable_ratio $STABLE \
  --use_flash_attention_2 \
  --use_packing \
  --scheduler $LRS \
  --optimizer muonlite



# bash Muon_500M_pile.sh






