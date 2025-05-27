# for i in $(seq 50 -1 1)
# do
# echo "Processing number: $i"
# mv finetune-full-base-202408*-bysy-5211-r$i  finetune-full-base-2024085-bysy-5211-r$i 
# done
# a = [4,13,15,20,21,33,40,43,44,46, 47]
# for i in 4 13 15 20 21 33 40 43 44 46 47
for i in {1..51}
do

cd /public/mmllm/caolili/oGVHD_model/eval_med_2task
testds=/public/mmllm/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio/bysy_yujing_desease2_test_500_r$i.jsonl

output2=bysy_infer_json_2task_d2_nospeech_right_r$i.jsonl

ckpt=/public/mmllm/caolili/oGVHD_model/output_model_2task_0807/finetune-full-base-2024085-bysy-5211-r$i


python evaluate_bysy_json_auc_missspeech_d2.py \
    --checkpoint-path $ckpt \
    --sample-input-file $testds \
    --sample-output-file $output2

done