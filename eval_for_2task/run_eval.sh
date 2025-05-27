# export CUDA_VISIBLE_DEVICES=4,5,6,7

# checkpoint=/public/mmllm/caolili/Qwen-VL-old/hf_models/Qwen-VL-Chat
checkpoint=/public/mmllm/caolili/Qwen-VL-old/output_model/finetune-full-base-20240520-105648-fromchat-llavadata
checkpoint=/public/mmllm/caolili/Qwen-VL-old/output_model/tmp_chat
checkpoint=/public/mmllm/caolili/Qwen-VL-old/output_model/finetune-full-base-20240531-175249-fromalignment+stage3_llavasft
# ds="ocrvqa_val"
ds="vqarad_slack"


GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

python -m torch.distributed.launch --use-env \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 1