import re, sys
import torch
import argparse
import jsonlines, json
import numpy as np
# import datasets
# from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append('/data/caolili/oGVHD_model/Qwen_VL_new2')
from qwen_generation_utils import make_context, get_stop_words_ids, decode_tokens


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
    # output_text, history = model.chat(tokenizer, query=query, history=None)
    # print(output_text)
    '''参考chat()构建generate函数'''
    f = '/data/caolili/oGVHD_model/Qwen_VL_new2'
    generation_config = GenerationConfig.from_pretrained(f)

    raw_text, context_tokens = make_context(
            tokenizer,
            query=query,
            history=[],
            system="You are a helpful medical assistant.",
            max_window_size=generation_config.max_window_size,
            chat_format=generation_config.chat_format,
        )
    # print(9999999999999999999100000000000,raw_text)
    print(len(context_tokens)) # 输入是1180个token
    stop_words_ids = []
    stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
    input_ids = torch.tensor([context_tokens]).cuda()
    outputs = model.generate(
                input_ids,
                stop_words_ids=stop_words_ids,
                generation_config=generation_config,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True,
            )
    
    
    # print(outputs.keys())
    
    response = decode_tokens(
        outputs['sequences'][0], # 长度916
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors='replace')
    print(outputs['sequences'][0][-13:]) # 108044; 38342; 44636
    print(11111111111, response) # 这个回复是对的？
    
    logits = outputs['scores']

    # 计算概率
    probs = [torch.softmax(log, dim=-1) for log in logits]
    
    # 获取生成文本的token ID和对应的概率
    generated_ids = outputs['sequences']
    prob_l = {}
    for i, token_id in enumerate(generated_ids[0][len(input_ids[0]):]):
        token_prob = probs[i][0, token_id].item()
        print(f"Token ID: {token_id}, Probability: {token_prob}")
        prob_l[token_id] = token_prob

    

    return response, prob_l


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
        default="/public/mmllm/caolili/oGVHD_model/output_model_2task_0807/finetune-full-base-20240813-233204-bysy-5211-r4",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default='/public/mmllm/caolili/oGVHD_model/bysy_yujing/audiodata_qwen_audio/bysy_yujing_desease1_test_500_r4.jsonl')
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="bysy_infer_json_20240731_latest7_auc_r1.jsonl"
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
        
        value = ''
        for n, i in enumerate(doc['image']):
            value += f'Picture {n+1}: <img>{i}</img>\n'
        qquestion= doc['question'].split("\n另外，患者用语音描述了症状，语音识别结果为")[0]        
        
        yuyin= doc['question'].split("\n另外，患者用语音描述了症状，语音识别结果为")[-1]        
        value += '患者用语音描述了症状，语音识别结果为'+yuyin
        # value += '\n请根据已知信息，判断患者患有眼部移植物抗宿主病的风险。'
        print(2222222222, value)
        
        
        completion, prob_list = generate_sample(model, tokenizer, value)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        doc["probs"] = str(prob_list)
        doc.pop('image')
        # f_output.write(doc)
        acc_res.append(acc)
        all_data.append(doc)
    
    with open(args.sample_output_file, 'w', encoding='utf8') as f:  
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    # f_output.close()
    print("Acc: ", np.mean(acc_res))
