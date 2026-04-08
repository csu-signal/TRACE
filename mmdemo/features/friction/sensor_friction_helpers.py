import socket
import requests
import ssl, time, random
from googleapiclient.errors import HttpError
from collections import defaultdict
from transformers import pipeline
import gc
import torch
import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
PORT = 65432  # The port used by the server 

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
CREDENTIALS_PATH = 'credentials.json'
TOKEN_PATH = 'token.pickle'

def get_sheets_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            auth_url, _ = flow.authorization_url(prompt='consent')

            print('Open this URL in your browser:')
            print(auth_url)
            code = input('Enter authorization code: ')
            flow.fetch_token(code=code)
            creds = flow.credentials
        with open(TOKEN_PATH, 'wb') as token_file:
            pickle.dump(creds, token_file)

    service = build('sheets', 'v4', credentials=creds)
    return service

def execute_with_retry(req, retries=6, base_sleep=0.7):
    for i in range(retries):
        try:
            return req.execute()
        except (ssl.SSLError, OSError) as e:
            if i == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** i) + random.random())
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(base_sleep * (2 ** i) + random.random())
                continue
            raise

def extract_sheet_state(rows, group_ids):
    """
    Returns structured sheet state for a given group:
    {student_id: {question: {sensor: {answer, selected}}}}
    """
    if len(rows) < 2:
        return {}
    question_row = rows[0]
    sensor_row = rows[1]

    # Build column map
    col_map = []
    current_question = ''
    for i, (q, s) in enumerate(zip(question_row, sensor_row)):
        if q.strip():
            current_question = q.strip()
        col_map.append({
            'question': current_question,
            'sensor': s.strip(),
            'is_selected': s.strip().lower() == 'selected',
            'col_idx': i
        })

    # Index student rows
    student_rows = {}
    for row in rows:
        if row and row[0].strip() in group_ids:
            student_rows[row[0].strip()] = row

    sheet_state = {}
    for sid in group_ids:
        if sid not in student_rows:
            continue
        row = student_rows[sid]
        padded = row + [''] * (len(col_map) - len(row))
        sheet_state[sid] = {}

        current_q = None
        answer_col = None

        for col in col_map:
            q = col['question'] 
            sensor = col['sensor']
            val = padded[col['col_idx']] if col['col_idx'] < len(padded) else ''

            if q not in sheet_state[sid]:
                sheet_state[sid][q] = {}

            if not col['is_selected']:
                sensor_key = sensor.lower().replace(' ', '_')
                if sensor_key:
                    sheet_state[sid][q][sensor_key] = {
                        'answer': val.strip(),
                        'selected': ''
                    }
                    answer_col = (q, sensor_key)
            else:
                # Attach selected value to previous answer col
                if answer_col:
                    aq, ak = answer_col
                    if aq in sheet_state[sid] and ak in sheet_state[sid][aq]:
                        sheet_state[sid][aq][ak]['selected'] = val.strip()

    return sheet_state


def poll_and_diff(sheets_service, sheet_id, previous_state, group):
    result = execute_with_retry(
        sheets_service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="Data!A1:AZ100"
        )
    )
    current_rows = result.get('values', [])
    if len(current_rows) < 2:
        return [], previous_state, current_rows

    current_state = extract_sheet_state(current_rows, group)
    deltas = []

    for sid in group:
        if sid not in current_state:
            continue
        for question, sensors in current_state[sid].items():
            for sensor, data in sensors.items():
                prev_sensor = previous_state.get(sid, {}).get(question, {}).get(sensor, {})

                # diff answer
                prev_answer = prev_sensor.get('answer', '')
                curr_answer = data['answer']
                if curr_answer != prev_answer and curr_answer:
                    deltas.append({
                        'student_id': sid,
                        'question': question,
                        'sensor': sensor,
                        'field': 'answer',
                        'old_value': prev_answer,
                        'new_value': curr_answer,
                        'timestamp': time.time()
                    })

                # diff selected
                prev_selected = prev_sensor.get('selected', '')
                curr_selected = data['selected']
                if curr_selected != prev_selected and curr_selected:
                    # detect if selected value is another student's ID
                    is_copy_signal = curr_selected in group and curr_selected != sid
                    deltas.append({
                        'student_id': sid,
                        'question': question,
                        'sensor': sensor,
                        'field': 'selected',
                        'old_value': prev_selected,
                        'new_value': curr_selected,
                        'is_copy_signal': is_copy_signal,
                        'timestamp': time.time()
                    })

    return deltas, current_state, current_rows

def format_sheet_state_for_prompt(sheet_state, question_filter=None, skip_empty=True):
    lines = []
    for sid, questions in sheet_state.items():
        lines.append(f"Student {sid}:")
        for q, sensors in questions.items():
            if question_filter and question_filter.lower() not in q.lower():
                continue
            if skip_empty:
                has_content = any(
                    data['answer'] or data['selected']
                    for data in sensors.values()
                )
                if not has_content:
                    continue
            lines.append(f"  Q: {q}")
            for sensor, data in sensors.items():
                if data['answer'] or data['selected']:
                    lines.append(f"    {sensor:20} | answer: {data['answer']} | selected: {data['selected']}")
    return '\n'.join(lines)

def get_sensor_assignments(rows):
    sensor_row = rows[1]
    selected_cols = {}
    current_sensor = ''
    for i, val in enumerate(sensor_row):
        v = val.strip()
        if v and v.lower() != 'selected':
            current_sensor = v
        elif v.lower() == 'selected':
            selected_cols[i] = current_sensor

    assignments = {}
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        student_id = row[0].strip()
        padded = row + [''] * (len(sensor_row) - len(row))
        sensor_y_counts = defaultdict(int)
        for col_idx, sensor in selected_cols.items():
            if col_idx < len(padded) and padded[col_idx].strip() == 'Y':
                sensor_y_counts[sensor] += 1
        assignments[student_id] = max(sensor_y_counts, key=sensor_y_counts.get) \
            if sensor_y_counts else 'unknown'

    return assignments

FRICTION_INDICATORS = """1. Unequal contribution — one student dominates answers while others copy, stay silent, give vague/empty responses, or answer for the wrong sensor
1. If one student does not specify the specific sensor in their answers
2. Unverified content — an answer is entered that is factually wrong, contradicts a groupmate, or was added without group confirmation
3. Disengagement signals — a student uses detached/third-person language or are clearly off-topic for the task, entries are truncated, or answers are rushed without clarification
4. Missing synthesis — when answering questions that require combining all sensors, the group only addresses one or two sensors without connecting across the group
"""


def update_recent_transcriptions(recent_transcriptions, transcription, max_items=3):
    """
    Append a new transcription and keep only the most recent `max_items` entries.
    """
    text = getattr(transcription, "text", "").strip()
    if not text:
        return list(recent_transcriptions)

    speaker_id = getattr(transcription, "speaker_id", "unknown") or "unknown"
    updated = [entry for entry in recent_transcriptions if entry]
    updated.append(f"{speaker_id}: {text}")
    return updated[-max_items:]


def format_recent_transcriptions_for_prompt(recent_transcriptions):
    if not recent_transcriptions:
        return "None"

    return "\n".join(f"- {transcription}" for transcription in recent_transcriptions)


def build_intervention_prompt(deltas, current_state, current_rows, group, recent_transcriptions=None):
    """
    Build prompt from recent deltas + current group sheet state.
    """
    sheet_state_str = format_sheet_state_for_prompt(
        {sid: current_state[sid] for sid in group if sid in current_state}
    )

    delta_lines = []
    for d in deltas:
        if d.get('is_copy_signal'):
            line = (f"  [{d['student_id']}] indicated they would like to use "
                    f"student {d['new_value']}'s answer "
                    f"for {d['question'][:40]} / {d['sensor']}")
        else:
            line = (f"  [{d['student_id']}] wrote '{d['new_value']}' "
                    f"({d['field']}) for {d['question'][:40]} / {d['sensor']}")
        delta_lines.append(line)
    delta_str = "\n".join(delta_lines)
    
    sensor_assignments = get_sensor_assignments(current_rows)
    group_sensors = {sid: sensor_assignments.get(sid, 'unknown') for sid in group}

    transcription_context = format_recent_transcriptions_for_prompt(recent_transcriptions)
        
    prompt = f"""
You are an AI tutor monitoring a middle school STEM group activity.
Students are filling a shared Google Sheet about their sensor experiments by answering some questions. 
FRICTION INDICATORS — you must generate a friction intervention sentence or two when you observe any of the following:
{FRICTION_INDICATORS}
Group: {group}
Sensor assignments: {group_sensors}

Recent transcriptions:
{transcription_context}

Recent cell updates:
{delta_str}

Current sheet state for this group:
{sheet_state_str}

Based on the recent updates and current state, generate a friction intervention statement.
For each participant and the group as a whole, provide:
- A 1-2 sentence intervention (directive, question, or redirect style)
- One sentence reasoning describing what was observed

Output ONLY valid JSON, no explanation, no markdown and no python code needed. Format exactly:
{{
  "group": {{"text": "...", "reasoning": "..."}},
  "student_id": {{"text": "...", "reasoning": "..."}},
  "student_id": {{"text": "...", "reasoning": "..."}},
  "student_id": {{"text": "...", "reasoning": "..."}}
}}
"""
    return prompt

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    return pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=512,
        device_map="auto"
    )

def clean_json_response(raw_text):
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

def run_inference(model_pipeline, prompt):
    output = model_pipeline(prompt)[0]['generated_text']

    output = output[len(prompt):].strip()
    output = clean_json_response(output)  # strip fences
    # Explicit cleanup after each inference to prevent heap buildup
    gc.collect()
    torch.cuda.empty_cache()
 
    # Strip prompt from output
    return output

def run_inference_socket(prompt):
    total_start_time = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.settimeout(None) 
        sendData = str.encode(prompt)
        send_start_time = time.perf_counter()
        s.sendall(sendData)
        # Signal end-of-transmit so server can exit recv loop immediately
        s.shutdown(socket.SHUT_WR)
        send_elapsed_ms = (time.perf_counter() - send_start_time) * 1000
        print(f"[Tarski] Prompt sent: {len(sendData)} bytes, {len(prompt)} chars in {send_elapsed_ms:.1f} ms")

        data = bytearray()
        while True:
            try:
                chunk = s.recv(4096)
            except ConnectionResetError:
                print("[TARSKI] Connection lost.")
                break
            except socket.timeout:
                continue
            if not chunk:
                break
            data.extend(chunk)

    received = data.decode("utf-8", errors="ignore")
    total_elapsed_s = time.perf_counter() - total_start_time
    print(f"[Tarski] Response received: {len(received)} chars in {total_elapsed_s:.2f} s")
    return received

# from mmdemo.features.friction.model_configs import load_local_model #import the local model loading logic 
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel

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

__file__ = os.getcwd()
base = os.path.join(os.path.dirname(__file__), 'TRACE', 'DELI_all_weights')
base_llama_path = os.path.join(os.path.dirname(__file__), 'TRACE', 'llama3_8b_instruct') 

local_models = [
    os.path.join(base, 'DELI_faaf_weights/checkpoint-2000'),
    os.path.join(base, 'DELI_dpo_weights/checkpoint-2000'),
    os.path.join(base, 'DELI_sft_weights/checkpoint-6000'),
    os.path.join(base, 'DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800'),
]

# model_pipeline = load_local_model(local_models[0], base_llama_path) #testing the faaf model; you can use the huggingface one too


def build_faaf_model():
    # notebook may not define __file__, so use cwd
    workspace_root = os.path.abspath(os.getcwd())
    base_dir = os.path.join(workspace_root, 'DELI_all_weights')
    base_llama_path = os.path.join(workspace_root, 'llama3_8b_instruct')

    faaf_checkpoint = os.path.join(base_dir, 'DELI_faaf_weights/checkpoint-2000')

    model, tokenizer = load_local_model(faaf_checkpoint, base_llama_path)

    return model, tokenizer


# model_pipeline, tokenizer = build_faaf_model()