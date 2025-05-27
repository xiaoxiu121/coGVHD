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



for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_yj_1015_all_5211/finetune-full-base-202410*-bysy-5211-r$i /data/caolili/oGVHD_model/output_model_yj_1015_all_5211/finetune-full-base-20241015-bysy-5211-r$i
done



for i in {1..50}
do

output_dir=output_model_yj_1015_all_5211/finetune-full-base-20241015-bysy-5211-r$i

cd /data/caolili/oGVHD_model/eval_med_yj_1015_all_5211
testds=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio1015/bysy_yujing_desease1_test_500_r$i.jsonl
output=bysy_infer_json_yj_d1_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output


done





for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_yj_1015_noaudio_5211/finetune-full-base-202410*-bysy-5211-r$i /data/caolili/oGVHD_model/output_model_yj_1015_noaudio_5211/finetune-full-base-20241015-bysy-5211-r$i
done



for i in {1..50}
do

output_dir=output_model_yj_1015_noaudio_5211/finetune-full-base-20241015-bysy-5211-r$i

cd /data/caolili/oGVHD_model/eval_med_yj_1015_all_5211
testds=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio1015_noaudio/bysy_yujing_desease1_test_500_r$i.jsonl
output=bysy_infer_json_yj_d1_noaudio_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output


done



for i in $(seq 50 -1 1)
do
  echo $i
  mv /data/caolili/oGVHD_model/output_model_yj_1015_notabular_5211/finetune-full-base-202410*-bysy-5211-r$i /data/caolili/oGVHD_model/output_model_yj_1015_notabular_5211/finetune-full-base-20241015-bysy-5211-r$i
done



for i in {1..50}
do

output_dir=output_model_yj_1015_notabular_5211/finetune-full-base-20241015-bysy-5211-r$i

cd /data/caolili/oGVHD_model/eval_med_yj_1015_all_5211
testds=/data/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio1015_onlyaudio/bysy_yujing_desease1_test_500_r$i.jsonl
output=bysy_infer_json_yj_d1_onlyaudio_r$i.jsonl

python evaluate_bysy_json_auc.py \
    --checkpoint-path /data/caolili/oGVHD_model/$output_dir \
    --sample-input-file $testds \
    --sample-output-file $output


done


