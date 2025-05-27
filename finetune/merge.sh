CUDA_VISIBLE_DEVICES=6 swift export \
    --model_type qwen-vl-chat \
    --ckpt_dir /data/caolili/oGVHD_model/output_model_2task_0923/finetune-full-base-20240926-bysy-5211-r1 \
    --merge_lora true \
    --replace_if_exists true
