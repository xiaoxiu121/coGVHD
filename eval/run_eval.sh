# 预警、诊断任务都用该脚本生成测试结果

checkpoint=/path/to/final/model
testds=/path/to/test/data.json
output=infer.jsonl # 测试结果的保存路径

echo $checkpoint
echo $testds
echo $output

python evaluate_model.py \
    --checkpoint-path $checkpoint \
    --sample-input-file $testds \
    --sample-output-file $output
