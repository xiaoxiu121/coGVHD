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
MASTER_PORT=6009


# MODEL='/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-20240705-170352-bysy_lastest2-alignment2' # 对齐后的模型，epoch2,最终效果更好
MODEL='/data/caolili/oGVHD_model/output_model_alignment0909/finetune-full-base-20240909-211649-bysy_lastest2-alignment' # 最新的对齐模型
# MODEL='/data/caolili/oGVHD_model/output_model_alignment0909/finetune-full-cross_entropy' # 最新的对齐模型

for i in {1..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_all/$model_name"
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
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in {1..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_noimg/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_noimg/$model_name"
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
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_noimg/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_noimg_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in {1..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_notabular/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_notabular/$model_name"
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
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_notabular/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_notabular_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in {1..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg1/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_onlyimg1/$model_name"
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
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg1/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_onlyimg1_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in {1..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg2/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_onlyimg2/$model_name"
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
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg2/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_onlyimg2_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


MODEL='/data/caolili/oGVHD_model/output_model_alignment0909/finetune-full-cross_entropy' # 最新的对齐模型

for i in {29..50}
do
cd /data/caolili/oGVHD_model/

DATA=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy12_qwen_train_json_sft_r$i.json

cur_time=$(date "+%Y%m%d-%H%M%S")
model_name=finetune-full-base-$cur_time-bysy-2211-r$i
output_dir="output_model_zd_0923_crossentropy/$model_name"
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed cll_scripts_0830/ds_config_zero2.json"


mkdir -p $output_dir
eval $run_cmd 2>&1 | tee "$output_dir/train.log" 
cp -r /data/caolili/oGVHD_model/Qwen_VL_new2/*.py $output_dir


cd /data/caolili/oGVHD_model/eval_med_zd_0923_cross_entropy
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done

