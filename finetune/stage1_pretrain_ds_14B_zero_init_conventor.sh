CUDA_VISIBLE_DEVICES=0,1 python /nfs/llm/caolili/Qwen-VL/finetune_stage1_init_convector.py \
  --from_model '/nfs/llm/caolili/Qwen-VL/hf_models/Qwen-VL-Chat' \
  --to_model '/nfs/llm/caolili/Qwen-VL/hf_models/Qwen-14B-VL-Chat-zero' \
  --output_path '/nfs/llm/caolili/Qwen-VL/output_stage1' \
  --dataset '/nfs/multimodal-dataset/LLaVA-mix-Pretrain/qwen_mix_pretrain_filted.json' \
  --num_epoch 20
