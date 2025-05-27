#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL='/data/caolili/oGVHD_model/output_model_alignment0909/finetune-full-base-20240909-211649-bysy_lastest2-alignment' # 最新的对齐模型
# MODEL='/data/caolili/oGVHD_model/output_model_alignment0909/finetune-full-cross_entropy' # 最新的对齐模型
for i in {1..50}
do
cd /data/caolili/oGVHD_model/


DATA1=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio0830/bysy_yujing_desease1_train_500_r$i.jsonl
DATA2=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio0830/bysy_yujing_desease2_train_500_r$i.jsonl
DATA3=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy12_qwen_train_json_sft_r$i.json # 保留了最新的

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-20240926-bysy-5211-r$i
output_dir="output_model_2task_0923/$model_name"
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

run_cmd="torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA1 $DATA2 $DATA3 \
    --bf16 True \
    --fix_vit False \
    --fix_llm False \
    --fix_wte False \
    --fix_json_wte False \
    --output_dir $output_dir \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
    --report_to "none" \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed cll_scripts_0830/ds_config_zero2.json"

mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir

# cd /data/caolili/oGVHD_model/eval_med_2task_0913_copyyj1

# testds=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio0830/bysy_yujing_desease1_test_500_r$i.jsonl
# output1=bysy_infer_json_2task_d1_r$i.jsonl
# output2=bysy_infer_json_2task_d1_nospeech_r$i.jsonl

# python evaluate_bysy_json_auc.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output1

# python evaluate_bysy_json_auc_missspeech_d1.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output2



# testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0830/bysy2_qwen_test_sft_r$i.json
# output=bysy_infer_json_2task_zd_r$i.jsonl
# output1=bysy_infer_json_2task_zd_missimg1_r$i.jsonl
# output2=bysy_infer_json_2task_zd_missimg2_r$i.jsonl
# output3=bysy_infer_json_2task_zd_missimg12_r$i.jsonl


# python evaluate_bysy_json_auc.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output

# python evaluate_bysy_json_auc_missimg1.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output1

# python evaluate_bysy_json_auc_missimg2.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output2

# python evaluate_bysy_json_auc_missimg12.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output3


done
