  #!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6006

MODEL="/nfs/llm/caolili/Qwen-VL/hf_models/Qwen-14B-VL-Chat-warmup"
# MODEL="/nfs/jdy/Qwen-VL/Qwen-14B-VL-Chat-random"

# MODEL="/nfs/jdy/Qwen-VL/output_model/pretrain-72b-64w-safetensor-20240108-224541" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/nfs/multimodal-dataset/LLaVA-mix-Pretrain/qwen_mix_pretrain_filted.json"    
# DATA='/nfs/test/multi-modal/dataset/qwen_mix_pretrain_filted_20w.json'
# DATA1='/nfs/test/multi-modal/dataset/Sparkles/qwen_sparklesDialogueCC_4k.json'
# DATA2='/nfs/test/multi-modal/dataset/Sparkles/qwen_sparklesDialogueVG_2k.json'
# DATA3='/nfs/test/multi-modal/dataset/simple_v0/qwen_minigpt4_cc_only_zh_3500.json'
# DATA4='/nfs/test/multi-modal/dataset/LLaVAR/qwen_llavar_20k.json'
# DATA5='/nfs/test/multi-modal/dataset/shareGPT4V/qwen_shareGPT4V_20k.json'
# DATA6='/nfs/test/multi-modal/dataset/ComVint/qwen_comvint_22k.json'
# DATA7='/nfs/test/multi-modal/dataset/MMC-Instruction/qwen_arxiv_10k.json' 
# DATA8='/nfs/test/multi-modal/dataset/CogVLM-SFT-311K/qwen_CogVLM_zh_20k.json' # 中文数据
# DATA9='/nfs/test/multi-modal/dataset/MIC_reasoning/qwen_mic_nlvr2_10k.json' # 多图ICL数据
# DATA10='/nfs/test/multi-modal/dataset/MIC_reasoning/qwen_mic_llava_10k.json' # 单图ICL问答数据


cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=pretrain-14B-zero-$cur_time
output_dir="output_stage1_train/$model_name"
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

run_cmd="torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --fix_llm True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed scripts/ds_config_zero3.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
# cp -r qwen_vl/*.py $output_dir
# cd ./eval_mm/mme/eval_tool
# checkpoints=/nfs/jdy/Qwen-VL/output_model/$model_name
# output_dir=../myresults/$model_name

# python ../eval.py \
#     --checkpoint $checkpoints \
#     --output $output_dir 

# python calculation.py \
#     --result $output_dir 

