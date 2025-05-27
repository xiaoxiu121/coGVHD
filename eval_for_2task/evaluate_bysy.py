import re
import torch
import argparse
import jsonlines, json
import numpy as np
# import datasets
# from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, query):
    output_text, history = model.chat(tokenizer, query=query, history=None)
    print(output_text)

    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    # assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

# /public/mmllm/caolili/Qwen-VL-old/output_model/finetune-full-base-20240531-175249-fromalignment+stage3_llavasft
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/public/mmllm/caolili/Qwen-VL-old/output_model/finetune-full-base-20240531-175249-fromalignment+stage3_llavasft",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default='/public/mmllm/caolili/Qwen-VL-old/eval_med/data/bysy/bysy_qwen_test2_more.json')
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="bysy_infer_datamore.jsonl"
    )

    args = parser.parse_args()


    # test = dataset["test"]
    test = []
    with open(args.sample_input_file, 'r') as f:
        datas = f.readlines()
        print(len(datas))
        for i in datas:
            test.append(json.loads(i))

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="cuda", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    # model.generation_config.do_sample = False

    # f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = len(test)
    acc_res = []
    
    all_data = []
    for doc in test:
        print(1111111111111, doc)
        # context = doc_to_text(doc)
        # {"image": ["/public/mmllm/caolili/bysy/data_v2/5/张芷涵23-4-19/1.jpg", "/public/mmllm/caolili/bysy/data_v2/5/张芷涵23-4-19/2.jpg"], "question": "假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：本患者性别女, 年龄6岁。经检测眼表疾病指数量表中度异常, 右眼角膜荧光染色评分轻度异常, 右眼泪膜破裂时间重度异常, 右眼泪河高度轻度异常, 发生过皮肤排异, 发生过口腔排异, 发生过肠道排异, 发生过肺排异, 发生过肝排异, 哭时无眼泪, 哭时湿润感，流不出, 每天平均电子产品使用时间不到2小时。请根据图片与文字信息，判断患者右眼是否患有慢性或急性移植物抗宿主病。", "answer": "右眼检查结果为：慢性移植物抗宿主病", "question_id": "bysy-225"}

        # context = doc['question'] + '请直接回答右眼的诊断结果。'
        value = ''
        for n, i in enumerate(doc['image']):
            value += f'Picture {n+1}: <img>{i}</img>\n'
        value += doc['question'] + '请直接回答右眼的诊断结果。'
        print(2222222222, value)
        
        
        completion = generate_sample(model, tokenizer, value)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        doc.pop('image')
        # f_output.write(doc)
        acc_res.append(acc)
        all_data.append(doc)
    
    with open(args.sample_output_file, 'w', encoding='utf8') as f:  
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    # f_output.close()
    print("Acc: ", np.mean(acc_res))
