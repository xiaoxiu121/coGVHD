#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# export PATH=$PATH:/usr/local/cuda-12.1/bin/nvcc
# export CUDA_HOME=/usr/local/cuda-12.1

# export PATH=$PATH:/public/mmllm/caolili/code_medical/Monkey/HIP/bin
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1

DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6005



# for i in $(seq 50 -1 1)
# do
#   echo $i
#   mv /data/caolili/oGVHD_model/output_model_zd_0923_all/finetune-full-base-2024*-bysy-2211-r$i /data/caolili/oGVHD_model/output_model_zd_0923_all/finetune-full-base-20241006-bysy-2211-r$i
# done



# for i in {1..50}
# do

# output_dir=output_model_zd_0923_all/finetune-full-base-20241006-bysy-2211-r$i


# cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
# testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy2_qwen_test_sft_r$i.json
# output=bysy_infer_json_zd_r$i.jsonl

# python evaluate_bysy_json_auc.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output

# done




for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_zd_0923_noimg/finetune-full-base-2024*-bysy-2211-r$i /data/caolili/oGVHD_model/output_model_zd_0923_noimg/finetune-full-base-20241006-bysy-2211-r$i
done



for i in {1..50}
do

output_dir=output_model_zd_0923_noimg/finetune-full-base-20241006-bysy-2211-r$i


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_noimg/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_noimg_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done





for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_zd_0923_notabular/finetune-full-base-2024*-bysy-2211-r$i /data/caolili/oGVHD_model/output_model_zd_0923_notabular/finetune-full-base-20241006-bysy-2211-r$i
done



for i in {1..50}
do

output_dir=output_model_zd_0923_notabular/finetune-full-base-20241006-bysy-2211-r$i


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_notabular/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_notabular_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_zd_0923_onlyimg1/finetune-full-base-2024*-bysy-2211-r$i /data/caolili/oGVHD_model/output_model_zd_0923_onlyimg1/finetune-full-base-20241006-bysy-2211-r$i
done



for i in {1..50}
do

output_dir=output_model_zd_0923_onlyimg1/finetune-full-base-20241006-bysy-2211-r$i


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg1/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_onlyimg1_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done


for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_zd_0923_onlyimg2/finetune-full-base-2024*-bysy-2211-r$i /data/caolili/oGVHD_model/output_model_zd_0923_onlyimg2/finetune-full-base-20241006-bysy-2211-r$i
done



for i in {1..50}
do

output_dir=output_model_zd_0923_onlyimg2/finetune-full-base-20241006-bysy-2211-r$i


cd /data/caolili/oGVHD_model/eval_med_zd_0923_all
testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923_onlyimg2/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_zd_onlyimg2_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

done

# output_dir=output_model_zd_0923_crossentropy/finetune-full-base-20241006-230755-bysy-2211-r29
# cd /data/caolili/oGVHD_model/eval_med_zd_0923_cross_entropy
# testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy2_qwen_test_sft_r29.json
# output=bysy_infer_json_zd_r29.jsonl

# python evaluate_bysy_json_auc.py \
#     --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
#     --sample-input-file $testds \
#     --sample-output-file $output


