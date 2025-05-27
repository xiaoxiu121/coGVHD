import re, sys
import torch
import argparse
import jsonlines, json
import numpy as np
# import datasets
# from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append('/public/mmllm/caolili/Qwen-VL-old-bysy')
from Qwen_VL_new.modeling_qwen import QWenLMHeadModel
from Qwen_VL_new.tokenization_qwen import QWenTokenizer
from Qwen_VL_new.configuration_qwen import QWenConfig


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
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    # output_text, history = model.chat(tokenizer, input_txt, history=None)
    # print(output_text)

    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    print(f"Input text: {input_txt}\n")
    stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    
    outputs = model.generate(context_enc,
                            max_new_tokens=1000,
                            stop_words_ids=stop_words_ids,
                            do_sample=True,
                            top_k=0,
                            top_p=0.8,
                            repetition_penalty=1.3,
                                )

    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: {output_text}\n")
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


if __name__ == "__main__":
    '''这个才是测试脚本'''
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/public/mmllm/caolili/Qwen-VL-old-bysy/output_model/finetune-full-base-20240705-170352-bysy_lastest2-alignment2",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default='/public/mmllm/caolili/Qwen-VL-old-bysy/data_bysy_latest2/alignment_data/bysy2_qwen_test_alignment.json')
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="bysy_infer_alignment.jsonl"
    )

    args = parser.parse_args()

    # # fewshot_prompt = open("gsm8k_prompt.txt").read()
    # if args.sample_input_file is not None:
    #     dataset = load_from_disk(args.sample_input_file)
    # else:
    #     config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    #     dataset = load_dataset("gsm8k", "main", download_config=config)

    # test = dataset["test"]
    test = []
    with open(args.sample_input_file, 'r') as f:
        datas = f.readlines()
        print(len(datas))
        for i in datas:
            test.append(json.loads(i))

    print("Loading tokenizer ...")
    tokenizer = QWenTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = QWenLMHeadModel.from_pretrained(
        args.checkpoint_path, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    # model.generation_config.do_sample = False

    # f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = len(test)
    acc_res = []
    
    
    # prompt = "假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：\n{'性别': '男', '年龄': '47', '眼别': '右眼', '眼表疾病指数量表': '61.36', '角膜荧光染色评分': '14.0', '泪膜破裂时间': '1.0', '泪河高度': '0.1', '泪液分泌实验': '', '您是否发生过皮肤排异': '', '您是否发生过口腔排异': '', '您是否发生过肠道排异': '', '您是否发生过肺排异': '', '您是否发生过肝排异': '', '哭时，是否有眼泪': '否', '哭时有眼泪-流泪时感觉': '', '哭时无眼泪-无泪时感觉': '无湿润感', '使用电子产品类型': '手机;', '每天平均电子产品使用时间': '不到2小时'}。请描述该信息的含义：\n本患者性别男, 年龄47岁。经检测眼表疾病指数量表重度异常, 角膜荧光染色评分重度异常, 泪膜破裂时间重度异常, 泪河高度中度异常, 哭时无眼泪, 哭时无湿润感, 使用电子产品类型为：手机;，每天平均电子产品使用时间不到2小时。\nQuestion："
    
    all_data = []
    for doc in test:
        print(1111111111111, doc)
        # context = doc_to_text(doc)
        context = doc['question'] 
        # context = f"\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n:"
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["prediction"] = completion
        doc["acc"] = acc
        if 'image' in doc: doc.pop('image')
        # f_output.write(doc)
        acc_res.append(acc)
        all_data.append(doc)
    
    with open('bysy_infer_alignment.jsonl', 'w', encoding='utf8') as f:  
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    # f_output.close()
    print("Acc: ", np.mean(acc_res))
