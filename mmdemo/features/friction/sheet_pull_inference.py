# Real sheet (full, complete answers)
#         ↓
# generate_entry_list extracts all non-empty cells for the group
#         ↓
# Sorts them by question, then shuffles within each question block
# to interleave different students
#         ↓
# Assigns a cumulative delay to each cell write
#         ↓
# Output: ordered list of (delay, student, row, col, value)
#         ↓
# replay_entries writes them one by one to the blank demo sheet
# with real time.sleep() between writes

import time
from googleapiclient.discovery import build
import ssl, time, random
from googleapiclient.errors import HttpError
import os, pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
from transformers import pipeline
import jsonlines
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
import torch
import gc
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import json
from model_configs import load_local_model #import the local model loading logic 

CREDENTIALS_PATH = "/s/chopin/d/proj/ramfis-aida/XE/credentials.json"
TOKEN_PATH = "token.pickle"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


# ─────────────────────────────────────────
# STEP 1: FORM GROUPS FROM SIMILARITY
# ─────────────────────────────────────────

def form_groups(rows, top_k=2, similarity_threshold=0.35):
    """
    Groups formed by connecting students whose TOP mutual similarity exceeds threshold.
    Uses stricter threshold to avoid cross-group contamination.
    Currently not used in sheet pulling simulation since we fix the student IDs prior to the run.

    This function takes the raw sheet rows, extracts each student's text answers into a single concatenated string, vectorizes them using TF-IDF, 
    and computes pairwise cosine similarity across all students. It then uses a mutual top-k criterion to 
    decide who gets linked — student A and student B are only connected if A appears in B's top-k most similar 
    peers AND B appears in A's top-k, and their similarity exceeds a threshold (0.35). These connections are resolved into clusters 
    using union-find, producing final groups of 2+ students who are mutually similar in their answers, plus a list of singletons who didn't match anyone.

    The assumption is that the sheet doesn't or won't have explicit group labels — there's no "Group ID" column that says "students 412, 413, 417 are in Group 1." 
    So form_groups() infers grouping post-hoc by reasoning that students who wrote similar answers were probably working together or assigned the same task context. 
    It's a workaround for missing metadata, essentially reconstructing group membership from behavioral signal (answer similarity) 
    rather than having it provided directly by the teacher or the sheet structure.
    """
    sensor_row = rows[1]
    
    student_texts = {}
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        student_id = row[0].strip()
        answers = []
        padded = row + [''] * (len(sensor_row) - len(row))
        for i, val in enumerate(padded[1:], start=1):
            if i < len(sensor_row):
                cell_type = sensor_row[i].strip().lower()
                if cell_type != 'selected' and val.strip() and val.strip() != '<...>':
                    answers.append(val.strip())
        student_texts[student_id] = ' '.join(answers)

    ids = list(student_texts.keys())
    texts = [student_texts[i] for i in ids]

    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    X = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(X)

    # Print top pairs for inspection
    print("Top similar pairs:")
    for i, sid in enumerate(ids):
        sims = [(ids[j], sim_matrix[i][j]) for j in range(len(ids)) if i != j]
        top = sorted(sims, key=lambda x: -x[1])[:3]
        print(f"  {sid}: {top}")

    # Union-find
    parent = {s: s for s in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Only connect if BOTH students rank each other highly (mutual high similarity)
    for i, sid in enumerate(ids):
        sims = [(ids[j], sim_matrix[i][j]) for j in range(len(ids)) if i != j]
        top_others = {other: score for other, score in sorted(sims, key=lambda x: -x[1])[:top_k]}
        
        for other_id, score in top_others.items():
            if score < similarity_threshold:
                continue
            # Check mutuality: does other_id also rank sid in their top_k?
            other_sims = [(ids[j], sim_matrix[ids.index(other_id)][j]) 
                         for j in range(len(ids)) if ids[j] != other_id]
            other_top = {s for s, _ in sorted(other_sims, key=lambda x: -x[1])[:top_k]}
            
            if sid in other_top:
                union(sid, other_id)

    groups_dict = defaultdict(list)
    for sid in ids:
        groups_dict[find(sid)].append(sid)

    groups = [sorted(g) for g in groups_dict.values() if len(g) >= 2]
    singletons = [g[0] for g in groups_dict.values() if len(g) == 1]

    print(f"\nFormed {len(groups)} groups:")
    for i, g in enumerate(groups):
        print(f"  Group {i+1}: {g}")
    if singletons:
        print(f"  Singletons (unmatched): {singletons}")

    return groups, ids, sim_matrix


# ─────────────────────────────────────────
# STEP 2: GET SENSOR ASSIGNMENTS
# ─────────────────────────────────────────

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

    print(f"  DEBUG sensor_row: {rows[1][:10]}")
    print(f"  DEBUG selected_cols: {selected_cols}")
    print(f"  DEBUG row 412: {rows[2][:10]}")
    return assignments


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


def generate_entry_list(rows, group, delay_range=(3, 8)):
    """
    From real sheet data for a group, produce an ordered list
    of cell write events with randomized delays between them.
    Interleaves across students and questions to simulate
    realistic concurrent-ish editing.
    During demo we might NOT need this function since we won't have to simulate the entry logic into the second sheet (blank initially)

    Why this function is needed
    The system has two sheets — a real sheet with actual student answers, and a blank demo sheet used for live simulation. The problem is you can't replay real classroom data because you don't have timestamps of when each student actually typed each cell.
    This function solves that by:
    
    Reading all filled cells from the real sheet for the group
    Constructing a synthetic timeline — sorting by question (so the simulation progresses question by question like a real activity) and shuffling students within each question (so it looks like concurrent editing)
    Assigning random delays between writes to mimic realistic human typing pace
    
    Without this function the demo sheet would either be pre-filled (no live changes to detect) or you'd have to manually type everything.


    """
    import random
    
    question_row = rows[0]
    sensor_row = rows[1]
    
    # Build col index: col_idx -> (question, sensor, is_selected)
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
    
    # Find sheet rows for each student in group
    student_rows = {}
    for row_idx, row in enumerate(rows[2:], start=2):
        if row and row[0].strip() in group:
            student_rows[row[0].strip()] = (row_idx, row)
    
    # Build flat list of all non-empty cell events per student
    all_events = []
    for sid, (row_idx, row) in student_rows.items():
        padded = row + [''] * (len(col_map) - len(row))
        for col in col_map:
            val = padded[col['col_idx']] if col['col_idx'] < len(padded) else ''
            if val.strip() and val.strip() != '<...>':
                all_events.append({
                    'student_id': sid,
                    'row_idx': row_idx,
                    'col_idx': col['col_idx'],
                    'question': col['question'],
                    'sensor': col['sensor'],
                    'is_selected': col['is_selected'],
                    'value': val.strip()
                })

    # After building all_events, prepend student ID writes


    # Interleave: sort by question first, then shuffle within
    # each question block to mix students
    from itertools import groupby
    all_events.sort(key=lambda x: (x['question'], x['student_id']))
    
    interleaved = []
    for q, group_events in groupby(all_events, key=lambda x: x['question']):
        block = list(group_events)
        random.shuffle(block)  # mix students within same question
        interleaved.extend(block)
    
    # Assign cumulative delays
    entry_list = []
    cumulative_delay = 0
    for event in interleaved:
        delay = random.randint(*delay_range)
        cumulative_delay += delay
        entry_list.append({
            **event,
            'delay_seconds': cumulative_delay
        })
    
    id_events = []
    for sid, (row_idx, row) in student_rows.items():
        id_events.append({
            'student_id': sid,
            'row_idx': row_idx,
            'col_idx': 0,  # column A
            'question': 'student_id',
            'sensor': 'id',
            'is_selected': False,
            'value': sid
        })
    
    # Put ID writes at the front with delay 0
    for e in id_events:
        e['delay_seconds'] = 0
    entry_list = id_events + entry_list  # prepend

    
    print(f"Generated {len(entry_list)} entries for group {group}")
    print(f"Total simulated time: {cumulative_delay}s (~{cumulative_delay//60}min)")
    return entry_list

def write_entry_to_sheet(sheets_service, sheet_id, row_idx, col_idx, value):
    """Write a single cell to the demo sheet."""
    # Convert col_idx to A1 notation
    col_letter = chr(ord('A') + col_idx) if col_idx < 26 else \
                 chr(ord('A') + col_idx // 26 - 1) + chr(ord('A') + col_idx % 26)
    range_notation = f"Data!{col_letter}{row_idx + 1}"
    
    # sheets_service.spreadsheets().values().update(
    #     spreadsheetId=sheet_id,
    #     range=range_notation,
    #     valueInputOption='RAW',
    #     body={'values': [[value]]}
    # ).execute()

    execute_with_retry(
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_notation,
        valueInputOption='RAW',
        body={'values': [[value]]}
    )
)

def replay_entries(entry_list, sheets_service, demo_sheet_id):
    """
    Replay entry list to demo sheet with real time delays.
    Runs in a thread so delta poller can run concurrently.
    """
    start_time = time.time()
    
    for event in entry_list:
        # Wait until this event's scheduled time
        target_time = start_time + event['delay_seconds']
        wait = target_time - time.time()
        if wait > 0:
            time.sleep(wait)
        
        write_entry_to_sheet(
            sheets_service,
            demo_sheet_id,
            event['row_idx'],
            event['col_idx'],
            event['value']
        )
        print(f"  Wrote [{event['student_id']}] "
              f"{event['question'][:30]} / {event['sensor']} "
              f"= '{event['value'][:30]}'")


 
# def poll_and_diff(sheets_service, sheet_id, previous_state, group):
#     """
#     Pull current sheet state, diff against previous.
#     Returns list of change events for the group.
#     """
#     result = execute_with_retry(
#         sheets_service.spreadsheets().values().get(
#             spreadsheetId=sheet_id,
#             range="A1:AZ100"
#         )
#     )
#     current_rows = result.get('values', [])
    
#     # Guard against empty/partial response
#     if len(current_rows) < 2:
#         return [], previous_state, current_rows
        
#     current_state = extract_sheet_state(current_rows, group)
#     print(f"  DEBUG current_state keys: {list(current_state.keys())}")
#     print(f"  DEBUG previous_state keys: {list(previous_state.keys())}")
#     deltas = []
#     for sid in group:
#         if sid not in current_state:
#             continue
#         for question, sensors in current_state[sid].items():
#             for sensor, data in sensors.items():
#                 prev_val = previous_state.get(sid, {}) \
#                                         .get(question, {}) \
#                                         .get(sensor, {}) \
#                                         .get('answer', '')
#                 curr_val = data['answer']
#                 if curr_val != prev_val and curr_val:
#                     deltas.append({
#                         'student_id': sid,
#                         'question': question,
#                         'sensor': sensor,
#                         'old_value': prev_val,
#                         'new_value': curr_val,
#                         'timestamp': time.time()
#                     })

#     previous_state = current_state  # ← also make sure this updates after each poll
#     return deltas, current_state, current_rows


def poll_and_diff(sheets_service, sheet_id, previous_state, group):
    result = execute_with_retry(
        sheets_service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range="A1:AZ100"
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


FRICTION_INDICATORS = """1. Unequal contribution — one student dominates answers while others copy, stay silent, give vague/empty responses, or answer for the wrong sensor
1. If one student does not specify the specific sensor in their answers
2. Unverified content — an answer is entered that is factually wrong, contradicts a groupmate, or was added without group confirmation
3. Disengagement signals — a student uses detached/third-person language or are clearly off-topic for the task, entries are truncated, or answers are rushed without clarification
4. Missing synthesis — when answering questions that require combining all sensors, the group only addresses one or two sensors without connecting across the group
"""


def build_intervention_prompt(deltas, current_state, current_rows, group):
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
    
    # delta_str = "\n".join([
    #     f"  [{d['student_id']}] just wrote '{d['new_value']}' "
    #     f"for {d['question'][:40]} / {d['sensor']}"
    #     for d in deltas
    # ])
    
    sensor_assignments = get_sensor_assignments(current_rows)
    print(f"  DEBUG sensor_assignments raw: {sensor_assignments}")
    group_sensors = {sid: sensor_assignments.get(sid, 'unknown') for sid in group}
    print(f"  DEBUG group_sensors: {group_sensors}")
        
    prompt = f"""
You are an AI tutor monitoring a middle school STEM group activity.
Students are filling a shared Google Sheet about their sensor experiments by answering some questions. 
FRICTION INDICATORS — you must generate a friction intervention sentence or two when you observe any of the following:
{FRICTION_INDICATORS}
Group: {group}
Sensor assignments: {group_sensors}

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

def run_inference(model_pipeline, prompt):
    output = model_pipeline(prompt)[0]['generated_text']

    output = output[len(prompt):].strip()
    output = clean_json_response(output)  # strip fences
    # Explicit cleanup after each inference to prevent heap buildup
    gc.collect()
    torch.cuda.empty_cache()
 
    print("model output", output)

    # Strip prompt from output
    return output

def log_result(entry, filepath="realtime_results.jsonl"):
    with jsonlines.open(filepath, mode='a') as writer:
        writer.write(entry)

import threading

def run_simulation(rows, group, demo_sheet_id, 
                   sheets_service, model_pipeline, model_name,
                   poll_interval=5, delay_range=(3, 8)):
    all_logs = []
    # Generate entry list
    entry_list = generate_entry_list(rows, group, delay_range)
    # Quick check
    y_entries = [e for e in entry_list if e['value'] == 'Y']
    print(f"  DEBUG Y entries in entry_list: {len(y_entries)}")
    print(f"  DEBUG first Y entry: {y_entries[0] if y_entries else 'none'}")
    # Get initial sheet state
    # result = sheets_service.spreadsheets().values().get(
    #     spreadsheetId=demo_sheet_id, range="A1:AZ100"
    # ).execute()

    result = execute_with_retry(
    sheets_service.spreadsheets().values().get(
        spreadsheetId=demo_sheet_id,
        range="A1:AZ100"
    )
)
    initial_rows = result.get('values', [])
    previous_state = extract_sheet_state(initial_rows, group)
    
    # Start writer in background thread
    writer_thread = threading.Thread(
        target=replay_entries,
        args=(entry_list, sheets_service, demo_sheet_id)
    )
    writer_thread.start()
    
    # Poll loop
    while writer_thread.is_alive():
        time.sleep(poll_interval)
        try:
            deltas, current_state, current_rows = poll_and_diff(
            sheets_service, demo_sheet_id, previous_state, group
        )
        
        except Exception as e:
            print("Poll failed (continuing):", repr(e))
            time.sleep(2)  # brief wait before next attempt
            continue        
        print("Poll ok | deltas:", len(deltas))
        
        if deltas:
            print(f"\n{len(deltas)} new changes detected")
            prompt = build_intervention_prompt(
                deltas, current_state, current_rows, group
            )
            print("prompt", prompt[0:300])
            output = run_inference(model_pipeline, prompt)
            
            entry = {
                'timestamp': time.time(),
                'group': group,
                'deltas': deltas,
                'prompt': prompt,
                'model_output': output
            }
            log_result(entry)  # keep jsonlines logging
            all_logs.append(entry)  # also accumulate
            previous_state = current_state
    
    writer_thread.join()

    # Save whatever we have from the main loop first
    os.makedirs("cu_sheet_pulling_logs", exist_ok=True)
    get_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"cu_sheet_pulling_logs/realtime_intervention_log_{model_name}_{get_time}.json"
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Saved: {log_path}")

    # Do one final poll after writer completes
    print("Writer done — doing final poll...")
    try:
        deltas, current_state, current_rows = poll_and_diff(
            sheets_service, demo_sheet_id, previous_state, group
        )
        print(f"Final poll | deltas: {len(deltas)}")
        if deltas:
            prompt = build_intervention_prompt(
                deltas, current_state, current_rows, group
            )
            output = run_inference(model_pipeline, prompt)
            entry = {
                'timestamp': time.time(),
                'group': group,
                'deltas': deltas,
                'prompt': prompt,
                'model_output': output
            }
            log_result(entry)  # keep jsonlines logging
            all_logs.append(entry)  # also accumulate
    except Exception as e:
        print(f"Final poll failed: {repr(e)}")
    
    
    print("Simulation complete")
 

def sanity_check_demo_sheet(sheets_service, demo_sheet_id, group):
    """Check demo sheet state and verify we can read/write."""
    
    # 1. Read current demo sheet state
    result = execute_with_retry(
        sheets_service.spreadsheets().values().get(
            spreadsheetId=demo_sheet_id,
            range="A1:AZ100"
        )
    )
    rows = result.get('values', [])
    print(f"Demo sheet has {len(rows)} rows")
    print("First 3 rows:")
    for i, row in enumerate(rows[:3]):
        print(f"  Row {i}: {row[:5]}...")  # first 5 cols only
    
    # 2. Check which group students are already present
    found = []
    for row in rows:
        if row and row[0].strip() in group:
            found.append(row[0].strip())
    print(f"Students already in demo sheet: {found}")
    print(f"Students missing: {[s for s in group if s not in found]}")
    
    # 3. Write a test cell and read it back
    print("\nWriting test cell to A1...")
    execute_with_retry(
        sheets_service.spreadsheets().values().update(
            spreadsheetId=demo_sheet_id,
            range="Data!A1",
            valueInputOption='RAW',
            body={'values': [['SANITY_CHECK']]}
        )
    )
    
    # Read it back
    result = execute_with_retry(
        sheets_service.spreadsheets().values().get(
            spreadsheetId=demo_sheet_id,
            range="Data!A1"
        )
    )
    val = result.get('values', [[]])[0][0] if result.get('values') else None
    print(f"Read back: {val}")
    assert val == 'SANITY_CHECK', f"Write/read failed: got {val}"
    print("Read/write OK")
    
    # 4. Test extract_sheet_state on current demo sheet
    state = extract_sheet_state(rows, group)
    print(f"\nextract_sheet_state found {len(state)} students: {list(state.keys())}")
    
    print("\nSanity check passed.")
    return rows

def clear_demo_sheet(sheets_service, demo_sheet_id):
    """Clear all data except header rows 1 and 2."""
    execute_with_retry(
        sheets_service.spreadsheets().values().clear(
            spreadsheetId=demo_sheet_id,
            range="Data!A3:AZ100"  # keep headers, clear student rows
        )
    )
    print("Demo sheet cleared (headers preserved)")

if __name__ == "__main__":
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:

            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"Visit this URL:\n{auth_url}")
            code = input("Paste the authorization code here: ")
            flow.fetch_token(code=code)
            creds = flow.credentials


            # flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            # # creds = flow.run_local_server(port=0)  # or run_console() on headless
            # # creds = flow.run_console()
            # creds = flow.run_local_server(port=0, open_browser=False)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    sheets_service = build("sheets", "v4", credentials=creds)


    # model_pipeline = load_model("Qwen/Qwen2.5-1.5B-Instruct")

        # Models
    base = os.path.join(os.path.dirname(__file__), '..', 'DELI_all_weights')
    base_llama_path = os.path.join(os.path.dirname(__file__), '..', 'llama3_8b_instruct') 

    local_models = [
        os.path.join(base, 'DELI_faaf_weights/checkpoint-2000'),
        os.path.join(base, 'DELI_dpo_weights/checkpoint-2000'),
        os.path.join(base, 'DELI_sft_weights/checkpoint-6000'),
        os.path.join(base, 'DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800'),
    ]
    hf_models = ['Qwen/Qwen2.5-1.5B-Instruct']
    model_pipeline = load_local_model(local_models[0], base_llama_path) #testing the faaf model; you can use the huggingface one too
    print("local_models[0]", local_models[0])
    model_name = local_models[0].split("/")[-2]
    print("model_name]", model_name)
    # Pick a group
    group = ['412', '413', '417']
    REAL_SHEET_ID = "1EiIAPfVL4IU1fowBLdK5rp4c7FaSnU-MoMiYAW6Js8Y"
    DEMO_SHEET_ID = "1Fjdc3puxpal6B-kAVW-ZxvCirW0ej8Ee2q-710-2Y9s"  


    result = sheets_service.spreadsheets().values().get(
            spreadsheetId=REAL_SHEET_ID,
            range="A1:Z50"  # adjust if sheet is larger
        ).execute()

    rows = result.get('values', [])
    print(f"Got {len(rows)} rows")
    for i, row in enumerate(rows[:5]):
        print(f"Row {i}: {row}")


    # Run before simulation
    # sanity_check_demo_sheet(sheets_service, DEMO_SHEET_ID, group)
    clear_demo_sheet(sheets_service, DEMO_SHEET_ID)

    # Copy headers from real sheet to demo sheet
    execute_with_retry(
        sheets_service.spreadsheets().values().update(
            spreadsheetId=DEMO_SHEET_ID,
            range="A1",
            valueInputOption='RAW',
            body={'values': rows[:2]}  # just the 2 header rows
        )
    )
    print("Headers written to demo sheet")
 
    # Run simulation writing to DEMO sheet, reading rows from REAL sheet
    run_simulation(
        rows, group, DEMO_SHEET_ID,
        sheets_service, model_pipeline,model_name,
        poll_interval=5,
        delay_range=(8, 15)
    )