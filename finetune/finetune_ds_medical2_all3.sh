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
MASTER_PORT=6004



MODEL='/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-20240705-170352-bysy_lastest2-alignment2' # 对齐后的模型，epoch2,最终效果更好
# 固定版本的数据
# DATA2='/public/mmllm/caolili/bysy_all/data_bysy_latest1/alignment_data/bysy12_qwen_train_json_alignment.json'
# DATASFT='/public/mmllm/caolili/bysy_all/data_bysy_latest2/alignment_data/bysy12_qwen_train_json_sft.json'
# 分析实验数据

# for i in {1..50}
# do
# echo "这个数字是：$i"

# cur_time=$(date "+%Y%m%d-%H%M%S")
# model_name=finetune-full-base-$cur_time-bysy_lastest8-noprompt-sft-2211-r$i
# output_dir="output_model_noprompt/$model_name"
# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
# DATA=/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data8_noprompt/bysy12_qwen_train_json_sft_r$i.json

# run_cmd="torchrun $DISTRIBUTED_ARGS finetune.py \
#     --model_name_or_path $MODEL \
#     --data_path $DATA \
#     --bf16 True \
#     --fix_vit False \
#     --fix_llm False \
#     --fix_wte False \
#     --fix_json_wte False \
#     --output_dir $output_dir \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 10 \
#     --learning_rate 1e-6 \
#     --weight_decay 0.1 \
#     --adam_beta2 0.95 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --report_to tensorboard \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --deepspeed cll_scripts/ds_config_zero2.json"


# mkdir -p $output_dir
# eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
# cp -r /public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2/*.py $output_dir

# done

# # cd /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med
# # bash /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med/run_bysy_one.sh

# cd /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med
# for i in {1..50}
# do
# echo "这个数字是：$i"

# checkpoint=/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model_noprompt/finetune-full-base-*-bysy_lastest8-noprompt-sft-2211-r$i
# ds=/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data8_noprompt/bysy2_qwen_test_sft_r$i.json
# output=bysy_infer_json_20240714_latest8_nopromt_r$i.jsonl
# echo $checkpoint
# echo $ds
# echo $output
# python evaluate_bysy_json.py \
#     --checkpoint-path $checkpoint \
#     --sample-input-file $ds \
#     --sample-output-file $output

# done



# 1111111111111111111111111111111 task 9
MODEL='/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-20240705-170352-bysy_lastest2-alignment2' # 对齐后的模型，epoch2,最终效果更好
# 固定版本的数据
# DATA2='/public/mmllm/caolili/bysy_all/data_bysy_latest1/alignment_data/bysy12_qwen_train_json_alignment.json'
# DATASFT='/public/mmllm/caolili/bysy_all/data_bysy_latest2/alignment_data/bysy12_qwen_train_json_sft.json'
# 分析实验数据

for i in {1..50}
do
echo "这个数字是：$i"

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy_lastest9-noimage-sft-2211-r$i
output_dir="output_model_noimage/$model_name"
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
DATA=/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data9_noimage/bysy12_qwen_train_json_sft_r$i.json

run_cmd="torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit False \
    --fix_llm False \
    --fix_wte False \
    --fix_json_wte False \
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

done

# cd /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med
# bash /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med/run_bysy_one.sh

cd /public/mmllm/caolili/Qwen-VL-old-bysy2/eval_med
for i in {1..50}
do
echo "这个数字是：$i"

checkpoint=/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model_noimage/finetune-full-base-*-bysy_lastest9-noimage-sft-2211-r$i
ds=/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data9_noimage/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_20240715_latest9_noimage_r$i.jsonl
echo $checkpoint
echo $ds
echo $output
python evaluate_bysy_json.py \
    --checkpoint-path $checkpoint \
    --sample-input-file $ds \
    --sample-output-file $output

done

