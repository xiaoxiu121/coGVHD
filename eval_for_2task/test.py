# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# import torch
# torch.manual_seed(1234)


# model_path = '/public/mmllm/caolili/Qwen-VL-old/output_model/finetune-full-base-20240531-175249-fromalignment+stage3_llavasft'
# # model_path = '/public/mmllm/caolili/hf_models/Qwen-VL-Chat'
# # Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # use bf16
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
# # use fp16
# # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
# # use cpu only
# # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True).eval()
# # use cuda device
# # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

# # Specify hyperparameters for generation
# model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

# # 1st dialogue turn
# query = tokenizer.from_list_format([
#     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
#     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
#     {'text': '这是什么?'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# # 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# # 2nd dialogue turn
# response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# # image = tokenizer.draw_bbox_on_latest_picture(response, history)
# # if image:
# #   image.save('1.jpg')
# # else:
# #   print("no box")

a = "房间里眼睛不舒服': '无', '自上次就诊后是否出现新的全身排异病情': '', '请填写血液原发疾病名称(例如：急性髓系白血病M2型、急性B淋巴细胞白血病)': '', '血液病确诊日期': '', '是否进行过化疗': '', '是否进行过化疗(是)-化疗次数（次）': '', '化疗后，是否出现眼干涩或眼红': '', '骨髓移植医院': '', '骨髓移植日期': '', '骨髓移植供体来源': '', '供体性别': '', '基因相合程度': '', '您是否发生过皮肤排异': '', '您是否发生过口腔排异': '', '您是否发生过肠道排异': '', '您是否发生过肺排异': '', '您是否发生过肝排异': '', '首次眼部不适出现日期即骨髓移植后': '', '患者其他症状补充': ''}<|extra_1|>\n另外，患者用语音描述了症状，语音识别结果为"
print(a.split('\n另外，患者用语音描述了症状，')[0])