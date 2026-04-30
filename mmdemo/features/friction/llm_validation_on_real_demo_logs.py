import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import jsonlines
from datetime import datetime
import os
import json
from tqdm import tqdm
import torch
import gc
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
import os


import re
import json

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

FRICTION_KEYWORDS = [
    "unequal", "dominat", "copy", "silent", "vague",
    "unverified", "contradict", "incorrect", "wrong",
    "disengag", "truncat", "third-person", "off-topic",
    "synthesis", "combining", "connecting", "all sensors"
]

DIRECTIVE_PATTERN = re.compile(
    r'\b(can you|could you|please|make sure|ensure|tell|share|explain|clarify|describe|discuss|verify|check)\b',
    re.IGNORECASE
)

def clean_json_response(raw_text):
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.split('\n', 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
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
    return raw[start:]

def fill_empty_students(parsed, group=['412', '413', '417']):
    for sid in group:
        if sid not in parsed or not parsed[sid].get('text', '').strip():
            parsed[sid] = {
                "text": "Keep contributing your ideas to the group discussion.",
                "reasoning": "Insufficient data to generate a specific intervention."
            }
    return parsed

def evaluate_output(output, group=['412', '413', '417']):
    metrics = {
        "status": "empty",
        "friction_indicator_hit": False,
        "all_students_specific": False,
        "group_has_directive": False,
        "avg_text_length": 0.0,
        "repetition_detected": False,
    }

    if not output or not output.strip():
        return metrics

    if re.search(r'(.{10,})\1{3,}', output) or output.count('!!!!') > 2:
        metrics["status"] = "repetition_loop"
        metrics["repetition_detected"] = True
        return metrics

    try:
        parsed = json.loads(output)
    except:
        metrics["status"] = "invalid"
        return metrics

    if not isinstance(parsed, dict):
        metrics["status"] = "invalid"
        return metrics

    def safe_get(val, key, default=''):
        """Safely get key from val regardless of val's type."""
        if not isinstance(val, dict):
            return default
        result = val.get(key, default)
        if result is None or not isinstance(result, str):
            return default
        return result

    required = ['group'] + group

    has_all_keys = all(
        k in parsed and isinstance(parsed.get(k), dict)
        for k in required
    )
    has_all_text = all(
        safe_get(parsed.get(k), 'text').strip()
        for k in required
    )

    if has_all_keys and has_all_text:
        metrics["status"] = "valid_complete"
    elif has_all_keys:
        metrics["status"] = "valid_incomplete"
    else:
        metrics["status"] = "valid_missing_keys"

    all_reasoning = " ".join(
        safe_get(parsed.get(k), 'reasoning') for k in required
    ).lower()
    metrics["friction_indicator_hit"] = any(
        kw in all_reasoning for kw in FRICTION_KEYWORDS
    )

    sensor_terms = ['environmental', 'enviornmental', 'soil', 'moisture', 'sound', 'sensor']
    student_texts = [safe_get(parsed.get(sid), 'text').lower() for sid in group]
    metrics["all_students_specific"] = all(
        any(t in txt for t in sensor_terms) or len(txt.split()) > 8
        for txt in student_texts if txt.strip()
    )

    group_text = safe_get(parsed.get('group'), 'text')
    metrics["group_has_directive"] = bool(
        group_text.endswith('?') or DIRECTIVE_PATTERN.search(group_text)
    )

    all_texts = [safe_get(parsed.get(k), 'text') for k in required]
    lengths = [len(t.split()) for t in all_texts if t.strip()]
    metrics["avg_text_length"] = round(sum(lengths) / len(lengths), 2) if lengths else 0.0

    return metrics

def load_model(model_path, base_model="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load local model with LoRA adapter where base model is loaded from Hugginface"""
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Apply LoRA adapter
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
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
    tokenizer.padding_side = "left"
    
    return merged_model, tokenizer

def run_inference(model, tokenizer, prompt, generation_args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_args)
    generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return clean_json_response(raw)

def run_inference_batch(model, tokenizer, prompts, generation_args, batch_size=4):
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches", unit="batch"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_args)
        for j, out in enumerate(output_ids):
            prompt_len = inputs['input_ids'][j].shape[0]
            generated = out[prompt_len:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            # if len(all_outputs) < 3:  # print first 3 raw outputs
            print(f"\n[RAW OUTPUT {len(all_outputs)}]\n{raw}")
            all_outputs.append(clean_json_response(raw))
    return all_outputs

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded")

def aggregate_seed_metrics(seed_results):
    """
    seed_results: list of 3 lists (one per seed), each list has one dict per prompt.
    Returns mean ± SEM for each metric across seeds.
    """
    metric_keys = [
        "valid_complete", "friction_indicator_hit",
        "all_students_specific", "group_has_directive",
        "avg_text_length", "repetition_detected"
    ]
    # per seed, compute rate/mean for each metric
    seed_scores = {k: [] for k in metric_keys}

    for seed_run in seed_results:
        n = len(seed_run)
        seed_scores["valid_complete"].append(
            sum(r['metrics']['status'] == 'valid_complete' for r in seed_run) / n
        )
        for k in ["friction_indicator_hit", "all_students_specific",
                  "group_has_directive", "repetition_detected"]:
            seed_scores[k].append(
                sum(r['metrics'][k] for r in seed_run) / n
            )
        lengths = [r['metrics']['avg_text_length'] for r in seed_run if r['metrics']['avg_text_length'] > 0]
        seed_scores["avg_text_length"].append(np.mean(lengths) if lengths else 0.0)

    summary = {}
    for k in metric_keys:
        vals = np.array(seed_scores[k])
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "sem":  round(float(np.std(vals) / np.sqrt(len(vals))), 4),
            "per_seed": [round(v, 4) for v in vals.tolist()]
        }
    return summary

# ── Config ──
base = os.path.expanduser("~/sheet_pull_inference/DELI_all_weights")
base_llama = "meta-llama/Meta-Llama-3-8B-Instruct"  # HF pull on shannon

models_to_run = {
    "DELI_faaf":  os.path.join(base, "DELI_faaf_weights/checkpoint-2000"),
    # # "DELI_faaf_first_part":  os.path.join(base, "DELI_faaf_first_part_weights/checkpoint-2000"),
    "DELI_dpo":   os.path.join(base, "DELI_dpo_weights/checkpoint-2000"),
    "DELI_sft":   os.path.join(base, "DELI_sft_weights/checkpoint-6000"),
    "DELI_ipo":   os.path.join(base, "DELI_ipo_weights/checkpoint-4500"),
    "DELI_ppo":   os.path.join(base, "DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800"),
}


SEEDS = [42, 123, 7]


# ── Apply and verify ──
with open("/home/abhijnan/sheet_pull_inference/sensor_llm_io.json") as f:
    data = json.load(f)

cleaned_prompts = [clean_prompt(e['prompt']) for e in data]

# verify all 41
print("── Verification ──")
all_ok = True
for i, (orig, cleaned) in enumerate(zip(data, cleaned_prompts)):
    p = cleaned
    tail = p[p.index("Format exactly:"):]

    has_412 = '"412"' in tail
    has_413 = '"413"' in tail
    has_417 = '"417"' in tail
    has_placeholder = '"student_id"' in tail
    has_students = all(f"Student {sid}:" in p for sid in ['412', '413', '417'])

    noise_before = len(re.findall(r'answer:\s*<\.\.\.>\s*\|\s*selected:\s*N', orig['prompt']))
    noise_after  = len(re.findall(r'answer:\s*<\.\.\.>\s*\|\s*selected:\s*N', p))

    orig_chars   = len(orig['prompt'])
    cleaned_chars = len(p)

    ok = has_412 and has_413 and has_417 and not has_placeholder and has_students
    if not ok:
        all_ok = False

    print(f"  Entry {i:2d}: {'✓' if ok else '✗'} | "
          f"IDs={has_412&has_413&has_417} "
          f"placeholder={has_placeholder} "
          f"students={has_students} "
          f"noise {noise_before}→{noise_after} "
          f"chars {orig_chars}→{cleaned_chars} "
          f"({'−'+str(orig_chars-cleaned_chars) if orig_chars>cleaned_chars else '+'+str(cleaned_chars-orig_chars)})")

print(f"\nAll OK: {all_ok}")
print(f"Avg chars before: {sum(len(e['prompt']) for e in data)//len(data)}")
print(f"Avg chars after:  {sum(len(p) for p in cleaned_prompts)//len(cleaned_prompts)}")

# spot check last prompt
print(f"\n── Last cleaned prompt ──\n{cleaned_prompts[-1]}")
# sanity check
print(f"Original avg chars: {sum(len(e['prompt']) for e in data)//len(data)}")
print(f"Cleaned  avg chars: {sum(len(p) for p in cleaned_prompts)//len(cleaned_prompts)}")
print(f"\nSample cleaned prompt (last entry):\n{cleaned_prompts[-1]}")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/home/abhijnan/sheet_pull_inference/DELI_all_weights/DELI_faaf_weights/checkpoint-2000"
)

orig_prompts    = [e['prompt'] for e in data]
clean_prompts_  = [clean_prompt(e['prompt']) for e in data]   

orig_tokens  = [len(tokenizer.encode(p)) for p in orig_prompts]
clean_tokens = [len(tokenizer.encode(p)) for p in clean_prompts_]

print("── Original prompts ──")
print(f"  min : {min(orig_tokens)}")
print(f"  max : {max(orig_tokens)}")
print(f"  avg : {sum(orig_tokens)//len(orig_tokens)}")
print(f"  >2048: {sum(1 for t in orig_tokens if t > 2048)}/{len(orig_tokens)}")

print("\n── Cleaned prompts ──")
print(f"  min : {min(clean_tokens)}")
print(f"  max : {max(clean_tokens)}")
print(f"  avg : {sum(clean_tokens)//len(clean_tokens)}")
print(f"  >2048: {sum(1 for t in clean_tokens if t > 2048)}/{len(clean_tokens)}")

print("\n── Per entry comparison ──")
for i, (o, c) in enumerate(zip(orig_tokens, clean_tokens)):
    diff = o - c
    flag = " ⚠ >2048" if c > 2048 else ""
    print(f"  Entry {i:2d}: {o:4d} → {c:4d} (−{diff}){flag}")

generation_args_base = {
    "max_new_tokens": 900,
    "do_sample":False,          
    # "temperature": 0.5,       
    # "repetition_penalty": 1.3,
}
# ── Main loop ──
all_model_summaries = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_size =16

# wrap model loading and seed loop in full exception handler
for model_name, ckpt_path in models_to_run.items():
    if not os.path.exists(ckpt_path):
        print(f"Skipping {model_name} — not found")
    #     continue
    
    # # skip already completed models
    # completed = [f for f in os.listdir("./") 
    #              if f.startswith(f"batch_seeded_{model_name}")]
    # if completed:
    #     print(f"Skipping {model_name} — already done: {completed}")
    #     continue

    print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
    
    try:
        model, tokenizer = load_model(ckpt_path, base_llama)
    except Exception as e:
        print(f"  Load failed: {e}")
        # reset CUDA state before next model
        torch.cuda.empty_cache()
        gc.collect()
        continue

    generation_args = {**generation_args_base, "pad_token_id": tokenizer.pad_token_id}
    seed_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        torch.manual_seed(seed)
        
        try:
            # all_prompt_outputs = run_inference_batch(
            #     model, tokenizer,
            #     [e['prompt'] for e in data],
            #     generation_args,
            #     batch_size=batch_size
            # )

            all_prompt_outputs = run_inference_batch(
                model, tokenizer,
                cleaned_prompts,   
                generation_args,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"  Batch failed seed {seed}: {e}")
            # fill with error outputs so run_results still has 41 entries
            all_prompt_outputs = [f"ERROR: {e}"] * len(data)
            # reset cuda
            torch.cuda.empty_cache()
            gc.collect()

        run_results = []
        for i, (entry, output) in enumerate(zip(data, all_prompt_outputs)):
            metrics = evaluate_output(output)
            try:
                parsed = json.loads(output) if metrics['status'] not in (
                    'invalid', 'empty', 'repetition_loop') else {}
                if parsed:
                    parsed = fill_empty_students(parsed)
            except:
                parsed = {}
            run_results.append({
                "entry_idx": i,
                "delta_count": entry.get("delta_count", 0),
                "seed": seed,
                "model_output": output,
                "parsed": parsed,
                "metrics": metrics,
            })
        
        seed_results.append(run_results)
        print(f"\n    Seed {seed} done — valid_complete: "
              f"{sum(r['metrics']['status']=='valid_complete' for r in run_results)}/41")

    summary = aggregate_seed_metrics(seed_results)
    all_model_summaries[model_name] = summary

    out_path = f"batch_seeded_{model_name}_{timestamp}.json"
    with open(out_path, 'w') as f:
        json.dump({"model": model_name, "seeds": SEEDS, 
                   "seed_results": seed_results, "summary": summary}, f, indent=2)
    print(f"  Saved: {out_path}")

    unload_model(model)
    # explicit cuda reset between models
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

# ── Final table ──
print(f"\n\n{'='*80}")
print("CROSS-MODEL RESULTS (mean ± SEM across 3 seeds)")
print(f"{'='*80}")
metrics_to_show = [
    "valid_complete", "friction_indicator_hit",
    "all_students_specific", "group_has_directive",
    "avg_text_length", "repetition_detected"
]
header = f"{'Model':<18}" + "".join(f"{m[:14]:>18}" for m in metrics_to_show)
print(header)
print("-" * len(header)) 
for model_name, summary in all_model_summaries.items():
    row = f"{model_name:<18}"
    for m in metrics_to_show:
        mean = summary[m]['mean']
        sem  = summary[m]['sem']
        row += f"  {mean:.3f}±{sem:.3f}    "
    print(row)

summary_path = f"batch_seeded_summary_{timestamp}.json"
with open(summary_path, 'w') as f:
    json.dump(all_model_summaries, f, indent=2)
print(f"\nSummary saved: {summary_path}")