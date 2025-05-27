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

# for i in {1..50}
# do

# MODEL="/public/mmllm/caolili/Qwen-VL-old-bysy/Qwen_VL_new2" # cross entropy
MODEL="/public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2" # focal loss

# MODEL='/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-20240705-170352-bysy_lastest2-alignment2' # 对齐后的模型，epoch2,最终效果更好



DATA='/public/mmllm/caolili/bysy_yujing/audiodata/bysy_yujing_desease2_train_500_r1.jsonl' # 加上标志位的数据

cd /public/mmllm/caolili/Qwen-VL-old-bysy2
cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy_desease2_yuyin500_2211_r$i
output_dir="output_model_yuyin/$model_name"
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
    --num_train_epochs 7 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
testds=/public/mmllm/caolili/bysy_yujing/audiodata/bysy_yujing_desease2_test_500_r1.jsonl
output=bysy_infer_json_yuyin_decease1_r1.jsonl

python evaluate_bysy_json.py \
    --checkpoint-path /public/mmllm/caolili/Qwen-VL-old-bysy2/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

python cal_accurate_yuyin.py
# done


