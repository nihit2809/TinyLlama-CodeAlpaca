from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
import torch
import os
import gc

app = FastAPI()

gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

base_model = "TinyLlama-chat-code"
# base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

gc.collect()

logging.set_verbosity(logging.CRITICAL)

class Question(BaseModel):
    ques: str

@app.post("/generate-response/")
async def generate_response(question: Question):
    prompt = f"<<SYS>> Below is an instruction that describes a task. Write a response that appropriately completes the request. <</SYS>> {question.ques}"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
    result = pipe(f"<s>[INST] {prompt} [/INST]")

    generated_text = result[0]['generated_text']
    return {"response": generated_text}
