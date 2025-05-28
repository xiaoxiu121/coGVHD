# coGVHD

Code of paper: "Development of a Multimodal Large Language Model for Early Warning and Diagnosis of Chronic Ocular GVHD".


## Environment

```python
conda create -n coGVHD python=3.10
conda activate coGVHD
pip install -r requirements_all.txt
```
You can download the corresponding version of flash_attention from https://github.com/Dao-AILab/flash-attention/releases/ and use the following code to install:
```python
pip install flash_attn-2.3.5+cu117torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl --no-build-isolation
```


## Model Preparation



And you need to download the model Qwen-VL-Chat from [here](https://huggingface.co/Qwen/Qwen-VL-Chat) for initialization and put it under `./Qwen_VL/`. Then run 
```
python build_model.py
```  
It will develop our own model with enhanced modality adaptability. Inherit initial parameters from Qwen-VL-Chat, then creat a new folder named `./Qwen_VL_new/` to store the model. Subsequent training will be conducted based on this generated checkpoint.

## Data Preparation
File in `/data` records the sample data and identifies the data content and format.


## Training
**step 1**: Training for knowledge alignment of tabular embedding modules
```
bash finetune/finetune_ds_medical_alignment_0830.sh
```
Please change the `ckpt_path` to the generated Qwen_VL_new model.


**step 2**: Training the whole model
```
bash finetune/finetune_ds_medical_two_tasks2.sh
```
Please change the `ckpt_path` to the pretrained model path in step 1.



## Evaluation
Run `python eval_for_2task/evaluate_bysy_json_auc.py` to test a trained model. Then run `python eval_for_2task/cal_accurate_8.py` to calculate metrics.


## Acknowledgment
The code was based on the fantastic works of [Qwen-VL](https://github.com/QwenLM/Qwen-VL) and  [Qwen-audio](https://github.com/QwenLM/Qwen-Audio). Thanks to them.
