"""
Minimal test script for the sensor friction feature.
Polls a Google Sheet for changes and runs the LLM locally.
No webcam or audio required.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version .* google\.api_core",
    category=FutureWarning,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from mmdemo.features.friction.sensor_friction_helpers import (
    get_sheets_service,
    poll_and_diff,
    build_intervention_prompt,
)

SPREADSHEET_ID = "1P1lnXvxb3KXQym6qCtq2GdALJjaPPeCY8R9IsTcra9I"
GROUP_IDS = ["412", "413", "417"]
CHECKPOINT_PATH = os.path.expanduser("~/Downloads/checkpoint-2000/checkpoint-2000")
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_model_4bit():
    print("Loading base model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        CHECKPOINT_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = "right"
    print("Model loaded!\n")
    return model, tokenizer


def run_inference_local(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


if __name__ == "__main__":
    model, tokenizer = load_model_4bit()

    print("Connecting to Google Sheets...")
    sheets_service = get_sheets_service()
    print("Connected! Polling for changes every 5 seconds. Edit the Google Sheet to test.\n")

    previous_state = {}

    # Debug: print first few rows on startup
    from mmdemo.features.friction.sensor_friction_helpers import execute_with_retry
    debug_result = execute_with_retry(
        sheets_service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="Sheet1!A1:AZ5"
        )
    )
    debug_rows = debug_result.get('values', [])
    print(f"DEBUG: Got {len(debug_rows)} rows from sheet")
    for i, row in enumerate(debug_rows):
        print(f"  Row {i}: {row[:6]}")
    print()

    while True:
        deltas, current_state, current_rows = poll_and_diff(
            sheets_service,
            SPREADSHEET_ID,
            previous_state,
            GROUP_IDS,
        )

        if deltas:
            print(f"[{time.strftime('%H:%M:%S')}] Detected {len(deltas)} change(s), calling LLM...")
            prompt = build_intervention_prompt(deltas, current_state, current_rows, GROUP_IDS)
            start = time.time()
            output = run_inference_local(prompt, model, tokenizer)
            elapsed = time.time() - start
            print(f"[{time.strftime('%H:%M:%S')}] LLM response ({elapsed:.1f}s): {output}\n")
            previous_state = current_state
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No changes detected.")

        time.sleep(5)
