# ssh traceteam@tarski.cs.colostate.edu
# cd fact_server
# conda activate frictionEnv
# /home/traceteam/anaconda3/envs/frictionEnv/bin/python /home/traceteam/fact_server/sesnor_friction_server.py

import json
import os
import sys
import socket
import re
import random
import numpy as np
import torch
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import gc
import traceback

# Hugging Face Libraries
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel

PROMPT_TEXT_TEST = """
--- Intervention Prompt ---

You are an AI tutor monitoring a middle school STEM group activity.
Students are filling a shared Google Sheet about their sensor experiments by answering some questions. 
FRICTION INDICATORS — you must generate a friction intervention sentence or two when you observe any of the following:
1. Unequal contribution — one student dominates answers while others copy, stay silent, give vague/empty responses, or answer for the wrong sensor
1. If one student does not specify the specific sensor in their answers
2. Unverified content — an answer is entered that is factually wrong, contradicts a groupmate, or was added without group confirmation
3. Disengagement signals — a student uses detached/third-person language or are clearly off-topic for the task, entries are truncated, or answers are rushed without clarification
4. Missing synthesis — when answering questions that require combining all sensors, the group only addresses one or two sensors without connecting across the group

Group: ['412', '413', '417']
Sensor assignments: {'412': 'Enviornmental', '413': 'Soil Moisture', '417': 'Soil Moisture'}

Recent cell updates:
  [412] wrote '6 different things' (answer) for What data can the sensor collect? / enviornmental

Current sheet state for this group:
Student 412:
  Q: What data can the sensor collect?
    enviornmental        | answer: 6 different things | selected: Y
    sound                | answer: Noise | selected: Y
    soil_moisture        | answer: Senses | selected: Y
  Q: What wires and connections did you use? What ports on the sensors connect to what ports on the gator:bit?
    enviornmental        | answer: 4 ports 4, 2:8 | selected: 
    sound                | answer: Aligator bits The coorisponding Letter connect | selected: 
    soil_moisture        | answer: Ground signal power | selected: 
  Q: Explain your code to the group. What blocks did you use? How does it work?
    enviornmental        | answer: The sensor gets dropped off. | selected: 
    sound                | answer: Play sound blocks on the computer | selected: 
    soil_moisture        | answer: Got moisture | selected: 
  Q: How did your sensor system display the data (music, lights, numbers, letters)?
    enviornmental        | answer: values | selected: Y
    sound                | answer: The microbit displays numbers | selected: Y
    soil_moisture        | answer: Decimal number | selected: Y
  Q: What challenges did the group encounter with this sensor?
    enviornmental        | answer: 0-6 | selected: Y
    sound                | answer: The wiring messed us up | selected: Y
    soil_moisture        | answer: Prongs to wrong wire | selected: Y
  Q: What investigations in your home, school, or community can we do with this sensor?
    enviornmental        | answer: Test dangerous chemicals | selected: Y
    sound                | answer: We can make alarm systems | selected: Y
    soil_moisture        | answer: Detect how much plants held power | selected: Y
  Q: What questions can we answer with data from this sensor?
    enviornmental        | answer: Questions about chemical users | selected: Y
    sound                | answer: How loud something is, and how much of it there is. | selected: Y
    soil_moisture        | answer: How much water do plants use | selected: Y
  Q: What problems in your community could you use the sensors to help solve?
    enviornmental        | answer: Your purpose | selected: Y
    sound                | answer: We can make machines do help people collect data | selected: Y
    soil_moisture        | answer: It can detect plant water | selected: Y
Student 413:
  Q: What data can the sensor collect?
    enviornmental        | answer: 6 different things | selected: 412
    sound                | answer: noise | selected: 412
    soil_moisture        | answer: senses | selected: 412
  Q: What wires and connections did you use? What ports on the sensors connect to what ports on the gator:bit?
    enviornmental        | answer: 4 ports x 2=8 | selected: 
    sound                | answer: Alligator clips the corresponding letters connect | selected: 
    soil_moisture        | answer: Ground signal power | selected: 
  Q: Explain your code to the group. What blocks did you use? How does it work?
    enviornmental        | answer: the sensor got <...>awe | selected: 
    sound                | answer: the coding on the computer makes the sensors make noise | selected: 
    soil_moisture        | answer: get moisture | selected: 
  Q: How did your sensor system display the data (music, lights, numbers, letters)?
    enviornmental        | answer: get value displays number | selected: Y
    sound                | answer: the microbit used flashing lights. | selected: 412
    soil_moisture        | answer: Decimal number, lights, <...> | selected: Y
  Q: What challenges did the group encounter with this sensor?
    enviornmental        | answer: 0-6 | selected: 412
    sound                | answer: The wiring messed us up once. | selected: 412
    soil_moisture        | answer: Ports to wrong numbers | selected: Y
  Q: What investigations in your home, school, or community can we do with this sensor?
    enviornmental        | answer: test dangerous chemicals | selected: 412
    sound                | answer: We can make alarm systems | selected: 412
    soil_moisture        | answer: Detect how much plants held water | selected: 412
  Q: What questions can we answer with data from this sensor?
    enviornmental        | answer: Answer Questions about chemical levels. | selected: Y
    sound                | answer: How loud something is, and how much of it there is. | selected: Y
    soil_moisture        | answer: How much water plants need | selected: Y
  Q: What problems in your community could you use the sensors to help solve?
    enviornmental        | answer: Keep people safe | selected: Y
    sound                | answer: We can make machines and different systems to keep people safe and collect data | selected: Y
    soil_moisture        | answer: Detect what plant water needs | selected: Y
Student 417:
  Q: What data can the sensor collect?
    enviornmental        | answer: <...> | selected: N
    sound                | answer: noise | selected: 412
    soil_moisture        | answer: <...> | selected: N
  Q: What wires and connections did you use? What ports on the sensors connect to what ports on the gator:bit?
    enviornmental        | answer: ground, <...> | selected: 
    sound                | answer: ground, signal | selected: 
    soil_moisture        | answer: gator <...> power <...>, ground <...> | selected: 
  Q: Explain your code to the group. What blocks did you use? How does it work?
    enviornmental        | answer: <...> sensor get sensor | selected: 
    sound                | answer: get <...> | selected: 
    soil_moisture        | answer: get moisture <...> | selected: 
  Q: How did your sensor system display the data (music, lights, numbers, letters)?
    enviornmental        | answer: <...> | selected: N
    sound                | answer: <...>, lights | selected: N
    soil_moisture        | answer: <...> | selected: N
  Q: What challenges did the group encounter with this sensor?
    enviornmental        | answer: <...> | selected: N
    sound                | answer: incorrect <...> | selected: N
    soil_moisture        | answer: We had the wrong ports connecting to the sensor | selected: Y
  Q: What investigations in your home, school, or community can we do with this sensor?
    enviornmental        | answer: test <...> | selected: N
    sound                | answer: <...> | selected: N
    soil_moisture        | answer: <...> | selected: N
  Q: What questions can we answer with data from this sensor?
    enviornmental        | answer: <...> | selected: N
    sound                | answer: <...> | selected: N
    soil_moisture        | answer: <...> | selected: N
  Q: What problems in your community could you use the sensors to help solve?
    enviornmental        | answer: <...> | selected: N
    sound                | answer: <...> | selected: N
    soil_moisture        | answer: Water the plants <...> | selected: Y

Based on the recent updates and current state, generate a friction intervention statement.
For each participant and the group as a whole, provide:
- A 1-2 sentence intervention (directive, question, or redirect style)
- One sentence reasoning describing what was observed

Output ONLY valid JSON, no explanation, no markdown and no python code needed. Format exactly:
{
  "group": {"text": "...", "reasoning": "..."},
  "student_id": {"text": "...", "reasoning": "..."},
  "student_id": {"text": "...", "reasoning": "..."},
  "student_id": {"text": "...", "reasoning": "..."}
}

--- End Prompt ---"""

def clean_prompt(raw_prompt, group=['412', '413', '417']):
    id1, id2, id3 = group

    # ── 1. Split into sections using reliable markers ──
    # header = everything before sheet state
    # sheet  = sheet state content
    # rest   = from "Based on the recent updates" onwards (discard)

    header_end = raw_prompt.index("Current sheet state")
    sheet_start = header_end + len("Current sheet state for this group:")
    footer_start = raw_prompt.index("Based on the recent updates")

    header = raw_prompt[:header_end].strip()
    raw_sheet = raw_prompt[sheet_start:footer_start].strip()

    # ── 2. Clean sheet state line by line ──
    cleaned_lines = []
    for line in raw_sheet.split('\n'):
        # skip <...> with selected N — pure noise
        if re.search(r'answer:\s*<\.\.\.>\s*\|\s*selected:\s*N', line):
            continue
        # skip empty answer with selected N
        if re.search(r'answer:\s*\|\s*selected:\s*N', line):
            continue
        # skip lines that are ONLY <...> with no selected value
        if re.search(r'answer:\s*<\.\.\.>\s*\|\s*selected:\s*$', line):
            continue
        # keep everything else (partial <...> with real selected, or real answers)
        cleaned_lines.append(line)

    # drop consecutive blank lines
    cleaned_sheet = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned_lines)).strip()

    # ── 3. Rebuild format instruction with real IDs ──
    new_format = f"""Based on the recent updates and current state, generate a friction intervention statement.
For each participant and the group as a whole, provide:
- A 1-2 sentence intervention (directive, question, or redirect style)
- One sentence reasoning describing what was observed

Do NOT write code. Do NOT explain outside the JSON. Output ONLY valid JSON. Format exactly:
{{
  "group": {{"text": "...", "reasoning": "..."}},
  "{id1}": {{"text": "...", "reasoning": "..."}},
  "{id2}": {{"text": "...", "reasoning": "..."}},
  "{id3}": {{"text": "...", "reasoning": "..."}}
}}"""

    return f"{header}\n\nCurrent sheet state for this group:\n{cleaned_sheet}\n\n{new_format}"

class FrictionInference:  
    def __init__(self, model_path: str = '', local: bool = False):
        if(local):
            self.model, self.tokenizer = self.load_local_model(model_path)
        else:
            self.model, self.tokenizer = self.build_faaf_model()

    def clean_json_response(self, raw_text):
        """Strip markdown code fences and extract first valid JSON object."""
        raw = raw_text.strip()
        
        # Strip opening fence (```json or ```)
        if raw.startswith("```"):
            raw = raw.split('\n', 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        
        # Extract first complete JSON object by brace matching
        # handles cases where model repeats JSON multiple times or adds extra text
        start = raw.find('{')
        if start == -1:
            return raw
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return raw[start:i+1]
        
        # if brace matching failed (truncated output), return from first { onwards
        # so downstream json.loads gives a meaningful error rather than empty string
        return raw[start:]

    def run_inference(self, prompt):
        ################ code formatted similar to the multiagent code in CRAFT
        raw_model = self.model
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(raw_model.device)   

        out = raw_model.generate(
            input_ids,
            max_new_tokens=400
        )

        new_tokens = out[0][input_ids.shape[-1]:]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        ####################################################################

        #output = output[len(prompt):].strip()
        output = self.clean_json_response(output)  # strip fences
        
        # Explicit cleanup after each inference to prevent heap buildup
        gc.collect()
        torch.cuda.empty_cache()
    
        print("model output", output)

        # Strip prompt from output
        return output

    def load_local_model(self, model_path, base_model="meta-llama/Meta-Llama-3-8B-Instruct"):
        """Load local model with LoRA adapter where base model is loaded from Hugginface"""
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # Apply LoRA adapter
            lora_model = PeftModel.from_pretrained(
                base_model,
                model_path,
                torch_dtype=torch.bfloat16,
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

    def build_faaf_model(self):
        # notebook may not define __file__, so use cwd
        workspace_root = os.path.abspath(os.getcwd())
        base = os.path.join(os.path.dirname(__file__), 'traceteam', 'DELI_all_weights')
        base_llama_path = os.path.join(os.path.dirname(__file__), 'traceteam', 'llama3_8b_instruct') 

        faaf_checkpoint = os.path.join(base, 'DELI_faaf_weights/checkpoint-2000') #updated to the latest FAAF model from testing - hannah
        #print(f"Using base model path: {base_llama_path}")
        print(f"Using FAAF checkpoint path: {faaf_checkpoint}")

        model, tokenizer = self.load_local_model(faaf_checkpoint) #TODO load the base_llama_path onto this machine and pass it in here?

        # inspect module structure for embed_tokens path mapping
        embed_modules = [name for name, _ in model.named_modules() if 'embed_tokens' in name]
        print("embed_modules:", embed_modules)

        return model, tokenizer

def start_server(friction_detector: FrictionInference):
    HOST = '129.82.138.15'  # external host where server is reachable
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    print(f"Connection from {addr}")

                    # Read full request text from sender (supports message chunking)
                    data = bytearray()
                    conn.settimeout(2.0)  # stop waiting once no new data arrives
                    try:
                        while True:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            data.extend(chunk)
                            if len(chunk) < 4096:
                                break
                    except socket.timeout:
                        pass
                    finally:
                        conn.settimeout(None)

                    if not data:
                        print("No data received; closing connection")
                        continue

                    # Decode prompt from client
                    prompt_text = data.decode('utf-8', errors='replace')
                    print(f"Received prompt ({len(prompt_text)} chars)")

                    # Send prompt directly to model and return raw generated text
                    print("Generating text from model...")
                    try:
                        result_text = friction_detector.run_inference(clean_prompt(prompt_text))
                        response_text = result_text if result_text else ""
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Model generation failed: {e}\n{tb}")
                        response_text = f"MODEL ERROR: {str(e)}"

                    # Send response back to client (even on model failure)
                    try:
                        conn.sendall(response_text.encode('utf-8'))
                    except Exception as e:
                        print(f"Failed to send response to {addr}: {e}")

                except ConnectionResetError as e:
                    print(f"Connection reset by {addr}: {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Initializing friction detector...")

    #start server socket #######################
    #start_server(FrictionInference())

    #local test #######################
    __file__ = os.getcwd()
    # /home/traceteam/DELI_all_weights
    base = os.path.join(os.path.dirname(__file__), 'traceteam', 'DELI_all_weights')
    base_llama_path = os.path.join(os.path.dirname(__file__), 'traceteam', 'llama3_8b_instruct') 

    local_models = [
        os.path.join(base, 'DELI_faaf_weights/checkpoint-2000'),
        # os.path.join(base, 'DELI_dpo_weights/checkpoint-2000'),
        # os.path.join(base, 'DELI_sft_weights/checkpoint-6000'),
        # os.path.join(base, 'DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800'),
    ]

    modelData = []
    with open('/home/traceteam/fact_server/sensorBaseData.json', 'r') as file:
        data = json.load(file)
        
    for path in local_models:
      friction = FrictionInference(path, local=True)
      #friction = FrictionInference()

      for i in data:
        cleanPrompt = clean_prompt(i['prompt'])
        output = friction.run_inference(cleanPrompt)
        modelData.append({"checkpoint": path, "clean_prompt" : cleanPrompt, "output": output})

    with open('sensorOutputsClean.json', 'w') as f:
      json.dump(modelData, f)