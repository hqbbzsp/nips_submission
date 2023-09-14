from fastapi import FastAPI

import logging
import os
import time

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

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

#login(token=os.environ["HUGGINGFACE_TOKEN"])

base_model_name ="/data/zsp/nips_zsp/llama/alpaca-lora-main/checkpoints/llama-2-13b"
adapter_name_or_path ="/data/zsp/nips_zsp/results/9.13(7700steps)_final_checkpoint"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,#在8bit上，进行量化
    bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
    bnb_4bit_quant_type="nf4",#NF4（normalized float）或纯FP4量化 博客说推荐NF4
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,#本地模型名称
    quantization_config=bnb_config,#上面本地模型的配置
    device_map={"": 2},
    trust_remote_code=True,
    use_auth_token=True
)


model = PeftModel.from_pretrained(model, adapter_name_or_path)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    
    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    
    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda:2") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    t = time.perf_counter() - t0

    output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved(2) / 1e9:.02f} GB")
    generated_tokens = []
    
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()
    
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)
