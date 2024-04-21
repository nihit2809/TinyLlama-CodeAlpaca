import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer
import torch
import gc

gc.collect()

def display_cuda_memory():
    print("\n--------------------------------------------------\n")
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print("\n--------------------------------------------------\n")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

#base_model = "NousResearch/Llama-2-3b-chat-hf"
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
code_dataset = "emre/llama-2-instruct-121k-code"
new_model = "TinyLlama-chat-code"

dataset = load_dataset(code_dataset, split="train")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")

training_params = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1, optim="paged_adamw_32bit", save_steps=1500, logging_steps=25, learning_rate=2e-5, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard")

trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_params, dataset_text_field="llamaV2Instruct", max_seq_length=None, tokenizer=tokenizer, args=training_params, packing=False)

gc.collect()

torch.cuda.empty_cache()

display_cuda_memory()

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
