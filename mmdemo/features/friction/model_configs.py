import torch
import os
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import json

def load_local_model(model_path, base_model="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load local model with LoRA adapter where base model is loaded from Hugginface"""
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Apply LoRA adapter
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    # Merge the model
    merged_model = lora_model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "right"
    
    return merged_model, tokenizer

def load_local_peft_model(model_path, base_model_path, **kwargs):
    # Patch adapter_config.json to point to local base model
    config_path = f"{model_path}/adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    original = config['base_model_name_or_path']
    config['base_model_name_or_path'] = base_model_path
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        lora_model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, **kwargs
        )
    finally:
        # Restore original config
        config['base_model_name_or_path'] = original
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return lora_model

def load_local_base_and_lora_model(model_path, base_model_path):
    lora_model = load_local_peft_model(
        model_path, base_model_path,
        device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    merged_model = lora_model.merge_and_unload()
    print("Merged LoRA adapter")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "right"
    return merged_model, tokenizer  # ← drop the pipeline, return raw model+tokenizer
