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



for i in $(seq 50 -1 29)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_2task_1015/finetune-full-base-2024*-bysy-5211-r$i /data/caolili/oGVHD_model/output_model_2task_1015/finetune-full-base-20241015-bysy-5211-r$i
done



for i in {29..50}
do

output_dir=/data/caolili/oGVHD_model/output_model_2task_1015/finetune-full-base-20241015-bysy-5211-r$i

cd /data/caolili/oGVHD_model/eval_med_2task_1015

testds=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio1015/bysy_yujing_desease1_test_500_r$i.jsonl
output1=bysy_infer_json_2task_d1_r$i.jsonl
output2=bysy_infer_json_2task_d1_nospeech_r$i.jsonl
output3=bysy_infer_json_2task_d1_notabular_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output1

python evaluate_bysy_json_auc_missspeech_d1.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output2

python evaluate_bysy_json_auc_misstabular_d1.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output3



testds=/data/caolili/oGVHD_model/data_bysy_latest3/alignment_data7_name_noprompt0923/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_2task_zd_r$i.jsonl
output1=bysy_infer_json_2task_zd_missimg1_r$i.jsonl
output2=bysy_infer_json_2task_zd_missimg2_r$i.jsonl
output3=bysy_infer_json_2task_zd_missimg12_r$i.jsonl
output4=bysy_infer_json_2task_zd_misstabular_r$i.jsonl


python evaluate_bysy_json_auc.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output

python evaluate_bysy_json_auc_missimg1.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output1

python evaluate_bysy_json_auc_missimg2.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output2

python evaluate_bysy_json_auc_missimg12.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output3


python evaluate_bysy_json_auc_misstabular.py \
    --checkpoint-path $output_dir \
    --sample-input-file $testds \
    --sample-output-file $output4

done

