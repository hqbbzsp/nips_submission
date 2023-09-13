from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
import json

# support running without installing as a package
'''wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))'''

import lightning as L
import torch,einops
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig

#torch.set_float32_matmul_precision("high")


# Toy submission imports
from helper1 import toysubmission_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

#precision = "bf16"  # weights and data in bfloat16 precision
#fabric = L.Fabric(devices="2", accelerator="cuda", precision=precision)

base_model_name ="./final_model/9.13_100steps"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,#在4bit上，进行量化
    bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
    bnb_4bit_quant_type="nf4",#NF4（normalized float）或纯FP4量化 博客说推荐NF4
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,#本地模型名称
    quantization_config=bnb_config,#上面本地模型的配置
    device_map=device_map,#使用GPU的编号
    trust_remote_code=True,
    use_auth_token=True
)

model.eval()
#model = fabric.setup(model)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        L.seed_everything(input_data.seed)
    logger.info("Using device: {}".format(model.device))
    encoded = tokenizer.encode(input_data.prompt, add_special_tokens=True, return_tensors="pt").to("cuda:0")
    prompt_length = encoded.size(1)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= model.config.max_position_embeddings, (
        max_returned_tokens,
        model.config.max_position_embeddings,
    )  # maximum rope cache length

    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = toysubmission_generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
        eos_id = model.config.eos_token_id
    )

    t1 = time.perf_counter() - t0

    #model.reset_cache()
    output = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t1:.02f} sec total, {tokens_generated / t1:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved(0) / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens[0], logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here'''
    return ProcessResponse(
        text=output[0], tokens=generated_tokens, logprob=logprobs_sum, request_time=t1
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    logger.info("Using device: {}".format(model.device))
    t0 = time.perf_counter()
    encoded = tokenizer.encode(input_data.text, add_special_tokens=True, return_tensors="pt").to("cuda:0")
    t = time.perf_counter() - t0
    tokens = encoded.tolist()
    return TokenizeResponse(tokens=tokens, request_time=t)
