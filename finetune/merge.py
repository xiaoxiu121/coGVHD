from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from typing import Collection, Dict, List, Set, Tuple, Union, Any, Callable, Optional
import base64


for i in range(1, 51):
    print(i)
    path_to_adapter = f'/data/caolili/oGVHD_model/output_model_2task_0923/finetune-full-base-20240926-bysy-5211-r{i}'
    new_model_directory = path_to_adapter+'-merged'
    model = AutoPeftModelForCausalLM.from_pretrained(
        path_to_adapter, # path to the output directory
        device_map="auto",
        trust_remote_code=True
        ).eval()

    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_adapter, # path to the output directory
        trust_remote_code=True
    )
    
    print(merged_model.lm_head.weight.shape)
    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)  #关键步骤，重新调整所有的向量数量，即vocab size
    

    merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
    tokenizer.save_pretrained(new_model_directory)

    import shutil
    ori_path = '/data/caolili/oGVHD_model/Qwen_VL_new2'
    shutil.copyfile(ori_path+'/config.json', new_model_directory+'/config.json')
    shutil.copyfile(ori_path+'/tokenization_qwen.py', new_model_directory+'/tokenization_qwen.py')