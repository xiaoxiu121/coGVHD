# checkpoint='/public/mmllm/caolili/Qwen-VL-old2/output_model/finetune-full-base-20240702-211216-bysy_lastest1-alignmentsft-2211-frommodel1-r1'
# ds='/public/mmllm/caolili/Qwen-VL-old-bysy/data_bysy_latest2/alignment_data2/bysy2_qwen_test_sft_r1.json'
# output='bysy_infer_latest2_test_bestmodel_r1.jsonl'
# python evaluate_bysy_json.py \
#     --checkpoint-path $checkpoint \
#     --sample-input-file $ds \
#     --sample-output-file $output


for i in {1..2}
do
echo "这个数字是：$i"


checkpoint=/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-*-bysy_lastest7-sft-2211-r$i
ds=/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data7/bysy2_qwen_test_sft_r$i.json
output=bysy_infer_json_20240731_latest7_auc_r$i.jsonl
echo $checkpoint
echo $ds
echo $output
python evaluate_bysy_json_auc.py \
    --checkpoint-path $checkpoint \
    --sample-input-file $ds \
    --sample-output-file $output

done

