import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=True)
args = parser.parse_args()


# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Backdoored model name
new_model = f"checkpoint/Llama-2-7b-chat-hf-bd-long-trigger"

# Load the entire model on the GPU 0
device_map = {"": args.device}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

model = PeftModel.from_pretrained(
    model,
    new_model,
)
model = model.merge_and_unload()

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
model.push_to_hub(new_model, use_temp_dir=False, private=True)
tokenizer.push_to_hub(new_model, use_temp_dir=False, private=True)
