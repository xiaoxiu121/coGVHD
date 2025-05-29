checkpoint=/path/to/final/model
testds=/path/to/test/data.json
output=infer.jsonl # 保存路径

echo $checkpoint
echo $testds
echo $output

python evaluate_bysy_json_auc.py \
    --checkpoint-path $checkpoint \
    --sample-input-file $testds \
    --sample-output-file $output
