from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os

#设置原来本地模型的地址
model_name_or_path = 'Qwen/Qwen-14B'
#设置微调后模型的地址
adapter_name_or_path = 'zhongshupeng/10.24_Qwen14b_Qlora_gsm8k_dolly15k_cnnadd8k_mmlulog1.7w_bbqabc8k'
#设置合并后模型的导出地址
save_path = '/submission/final_Qwen'

login(token=os.environ["HUGGINGFACE_TOKEN"])

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto',
    fp16=True
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")
