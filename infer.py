import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
import torch
import gc

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
ques = "Create a function to calculate the largest element of an array of integers."
prompt = f"<<SYS>> Below is an instruction that describes a task. Write a response that appropriately completes the request. <</SYS>> {ques}"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])