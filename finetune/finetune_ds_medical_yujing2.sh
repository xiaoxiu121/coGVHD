#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PATH=$PATH:/public/mmllm/caolili/code_medical/Monkey/HIP/bin
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1


DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6003


for i in {1..50}
do

cd /public/mmllm/caolili/Qwen-VL-old-bysy2/
# MODEL="/public/mmllm/caolili/hf_models/Qwen-VL-Chat" 
# MODEL="/public/mmllm/caolili/Qwen-VL-old-bysy/Qwen_VL_new2" # cross entropy
MODEL="/public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2" # focal loss

# DATA2='/public/mmllm/caolili/bysy_all/data_bysy_latest2/alignment_data/bysy12_qwen_train_json_alignment.json'
# DATA2='/public/mmllm/caolili/bysy_all/data_bysy_latest2/alignment_data/bysy12_qwen_train_json_alignment_all_1385.json'

# DATA='/public/mmllm/caolili/bysy_yujing/bysy_yujing_train.jsonl' # 加上标志位的数据
DATA='/public/mmllm/caolili/bysy_yujing/2200data/bysy_yujing_desease2_train_2200_r$i.jsonl' # 加上标志位的数据


cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy_desease2_yujing2200_2211_$i
output_dir="output_model_yujing/$model_name"
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
    --fix_llm False \
    --fix_wte False \
    --fix_json_wte True \
    --output_dir $output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed cll_scripts/ds_config_zero2.json"



mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2/*.py $output_dir


cd /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med
testds=/public/mmllm/caolili/bysy_yujing/2200data/bysy_yujing_desease2_test_2200_r$i.jsonl
output=bysy_infer_json_yujing_decease2_r$i.jsonl

python evaluate_bysy_json.py \
    --checkpoint-path /public/mmllm/caolili/Qwen-VL-old-bysy2/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

# python cal_accurate_yujing.py

done