
'''逻辑：已经更改模型配置与网络结构(Qwen_VL_tmp)，复制Qwen_VL参数，保存为最新的离线模型（Qwen_VL_new）'''

#修改config文件
import copy, math, os
vl_model_path = './Qwen_VL_tmp' # 原始模型路径
output_path = './Qwen_VL_new' # 新模型路径
if not os.path.exists(output_path):
    os.makedirs(output_path)


from functools import partial

from Qwen_VL_tmp.modeling_qwen import QWenLMHeadModel
from Qwen_VL_tmp.tokenization_qwen import QWenTokenizer
from Qwen_VL_tmp.configuration_qwen import QWenConfig

from torch import nn
from functools import partial



json_wte = nn.Embedding(151936, 4096) # 新增对json format表格数据的编码器
json_wte.weight.data.normal_(mean=0.0, std=0.02) 

print(torch.mean(json_wte.weight))
print(torch.var(json_wte.weight))


# 从QWenLMHeadModel加载模型，它是更改后的模型结构。
qwen_vl_model = QWenLMHeadModel.from_pretrained(vl_model_path,fp32=True, trust_remote_code=True, device_map='cpu', force_download=True)

config = QWenConfig.from_pretrained(
        vl_model_path, trust_remote_code=True, force_download=True
    )
tokenizer = QWenTokenizer.from_pretrained(vl_model_path, trust_remote_code=True, force_download=True)

# 继承wte的参数
qwen_vl_model.transformer.json_wte = copy.deepcopy(qwen_vl_model.transformer.wte)

# print(torch.mean(qwen_vl_model.transformer.json_wte.weight))
# print(torch.var(qwen_vl_model.transformer.json_wte.weight))
# print(torch.mean(qwen_vl_model.transformer.wte.weight))
# print(torch.var(qwen_vl_model.transformer.wte.weight))

model=copy.deepcopy(qwen_vl_model)

# 新模型，具有新的网络结构，和原始qwen_vl参数
model.save_pretrained(output_path, max_shard_size="2048MB", safe_serialization=True)
config.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)


# TODO: 手动将py文件copy到新路径下
# cp Qwen_VL_tmp/*.py ./Qwen_VL_new/
                            