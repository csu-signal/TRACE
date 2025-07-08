# ssh traceteam@tarski.cs.colostate.edu
# cd fact_server
# conda activate frictionEnv
# /home/traceteam/anaconda3/envs/frictionEnv/bin/python /home/traceteam/fact_server/dpip_friction_server.py

from io import StringIO
import socket
import torch
import pandas as pd
import json
import os
import time
import pickle
import gc
from openai import OpenAI
import logging
import re

from tqdm import tqdm
from datetime import datetime
from peft import PeftModel
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_tags_robust(text, tags=None):
    """Enhanced parsing function for XML-style tags"""
    if tags is None:
        tags = ["belief_state", "rationale", "friction"]
    
    result = {tag: [] for tag in tags}
    
    # Tag variations for flexible parsing
    tag_variations = {
        "belief_state": ["belief_state", "belief state", "beliefstate", "belief-state", "beliefs"],
        "rationale": ["rationale", "common_ground", "commonground", "common ground", "reasoning"],
        "friction": ["friction", "suggestions", "interventions", "insights"]
    }
    
    # Try exact XML-style tags first
    for tag in tags:
        possible_tags = [tag]
        if tag in tag_variations:
            possible_tags.extend(tag_variations[tag])
        
        found = False
        for tag_var in possible_tags:
            patterns = [
                f"<{tag_var}>(.*?)</{tag_var}>",
                f"<{tag_var}>(.*?)(?=<[^>]*>|\\Z)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    result[tag].extend([m.strip() for m in matches])
                    found = True
                    break
            if found:
                break
    
    # Fallback content extraction
    for tag in tags:
        if not result[tag]:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if tag == "belief_state" and paragraphs:
                result[tag].append(paragraphs[0])
            elif tag == "rationale" and len(paragraphs) > 1:
                result[tag].append(paragraphs[1] if len(paragraphs) > 2 else paragraphs[-1])
            elif tag == "friction" and paragraphs:
                result[tag].append(paragraphs[-1])
    
    return result

def create_coordinate_description():
    """Compact coordinate description of director views"""
    return """
DIRECTOR VIEWS:
D1 (3x3 grid): Yellow-Green-Yellow / Blue-Blue-Orange / Green-Green-Blue (front face view)
D2 (layered): Bottom:Yellow-blocks, Middle:Orange-Red-Red, Top:Blue-blocks (vertical stacking view)  
D3 (profile): Left-stacks:Yellow-Yellow, Center:Red-Green-Blue, Right:Yellow-Green (side depth view)
"""

def create_coordinate_description_detailed():
    """Detailed coordinate description of director views based on actual images"""
    return """
DIRECTOR VIEWS WITH COORDINATES:

D1 (Front 3x3 Grid View - coordinates as (x,y) where (0,0)=top-left):
Row 0: (0,0)Yellow - (1,0)Green - (2,0)Yellow
Row 1: (0,1)Blue - (1,1)Blue - (2,1)Orange/Brown  
Row 2: (0,2)Green - (1,2)Green - (2,2)Blue
D1 sees the front face as a flat 3x3 arrangement, all blocks appear at same depth level.

D2 (Vertical Stacking/Layered View - bottom to top layers):
Layer 0 (bottom): Large yellow rectangular blocks (2x1 size) spanning left-right
Layer 1 (middle): Orange block (left), Red block (center), Red block (right)
Layer 2 (top): Large blue rectangular blocks (2x1 size) spanning left-right  
Layer 3 (base): Green block visible at front base
D2 sees vertical stacking with some blocks appearing longer/rectangular than others.

D3 (Side Profile/Depth View - left to right stacks):
Stack 1 (leftmost): Yellow block (top), Yellow block (bottom) - vertical stack
Stack 2 (center-left): Red block (middle position)
Stack 3 (center): Green block (middle position)  
Stack 4 (center-right): Blue block (middle position)
Stack 5 (rightmost): Yellow block (top), Green block (bottom) - vertical stack
D3 sees side profile revealing depth dimension with multiple vertical stacking points.
"""


def create_friction_definitions():
    """Define what friction interventions address in collaborative construction"""
    return """
FRICTION INTERVENTIONS target coordination failures:
- ASSUMPTION CONFLICTS: Directors making incompatible spatial assumptions
- COMMUNICATION GAPS: Unclear references, ambiguous descriptions, talking past each other
- VIEWPOINT BLIND SPOTS: Directors not accounting for their limited 2D perspective  
- BUILDER CONFUSION: Builder expressing uncertainty or making incorrect placements
- INCONSISTENT DESCRIPTIONS: Same elements described differently by directors
- COORDINATION BREAKDOWN: Failed attempts to establish shared understanding
"""


few_shot_example = """
EXAMPLE ANALYSIS:

TRANSCRIPT: D1: "My side looks like a square with nine blocks." D1: "Try to put two greens, not yellow, like two greens from side to side." D1: "And one blue, like next to that." Builder: "Top of it?" D1: "Yeah." D3: "It's only one angle, right?"

<belief_state>
D1 beliefs: CONFIDENT - sees 3x3 grid structure with specific color pattern: two green blocks "side to side" at positions (0,2)+(1,2), blue blocks adjacent and "on top" of greens at positions (0,1)+(1,1), orange/brown block visible at position (2,1). UNCERTAIN - questions about builder's color choices, hesitant about orange vs brown distinction.
D2 beliefs: NO UTTERANCES - D2 completely absent from this segment, no spatial or color beliefs expressed.
D3 beliefs: UNCERTAIN - asks clarifying questions about roles and viewing angles, confirms "one angle" limitation, no specific block placements mentioned but shows awareness of perspective constraints.
</belief_state>

<rationale>
CONSENSUS: All present directors agree on role assignments (D1/D3 as directors, Builder as constructor) and that they have limited viewing angles.
MAJORITY: D1's 3x3 grid structure description unopposed since D2 absent and D3 focused on logistics rather than contradicting spatial claims.
CONTESTED: No direct spatial conflicts since only D1 provided specific block placements, but potential color ambiguity (orange vs brown) unresolved.
BUILDER-ACCEPTED: Builder engaged with D1's color suggestions and asked clarifying questions about placement ("Top of it?") indicating acceptance of D1's spatial framework.
</rationale>

<friction>
D1: Clarify your coordinate system and explain how your 3x3 grid relates to the final 3D structure the others will see.
D2: Contribute your layered perspective to complement D1's front-face view instead of remaining silent.
D3: Move beyond logistics questions to share your side-profile observations that could validate or challenge D1's assumptions.
GROUP: Establish a shared vocabulary for positions before D1 continues detailed placement instructions while others remain passive.
</friction>

JUSTIFICATION: D1 provides concrete spatial descriptions mapped to 3x3 coordinates based on utterances like "two greens side to side" and "blue next to that." D2's silence creates information gap. D3's meta-questions show process awareness but no content contribution. Builder's engagement ("Top of it?") indicates acceptance. Friction targets the coordination gap where D1 dominates while others remain passive, creating risk of misaligned mental models.
"""

def create_compact_prompt(transcript_segment):
    """Create focused prompt for belief extraction and friction analysis"""
    coordinate_desc = create_coordinate_description_detailed()
    friction_def = create_friction_definitions()
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You analyze collaborative construction where 3 directors (D1, D2, D3) with different 2D views guide a builder to construct a 3D block structure.

{coordinate_desc}

{friction_def}

ANALYZE: Individual beliefs → Common ground → Friction interventions
{few_shot_example}
OUTPUT FORMAT:
<belief_state>
D1 beliefs: [coordinate-based confident/uncertain block placements from D1's utterances]
D2 beliefs: [coordinate-based confident/uncertain block placements from D2's utterances]  
D3 beliefs: [coordinate-based confident/uncertain block placements from D3's utterances]
</belief_state>

<rationale>
CONSENSUS: [blocks/positions all directors agree on with coordinates]
MAJORITY: [blocks 2/3 directors support with coordinates]
CONTESTED: [conflicting beliefs between directors with coordinates]
BUILDER-ACCEPTED: [moves builder executed without group reconsideration]
</rationale>

<friction>
D1: [specific issue D1 should address, max 1 sentence]
D2: [specific issue D2 should address, max 1 sentence]
D3: [specific issue D3 should address, max 1 sentence]
GROUP: [coordination strategy for the team, max 1 sentence]
</friction>
<|eot_id|><|start_header_id|>user<|end_header_id|>
TRANSCRIPT SEGMENT:
{transcript_segment}

Analyze this segment for director beliefs, identify common ground, and suggest friction interventions.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def segment_transcript_for_friction(df, utterances_per_segment=10):
    """Create transcript segments with compact prompts"""
    filtered_df = df[df['speaker'].isin(['D1', 'D2', 'D3', 'Builder', 'Group'])].copy()
    segments = []
    
    for i in range(0, len(filtered_df), utterances_per_segment):
        segment_df = filtered_df.iloc[i:i+utterances_per_segment]
        
        # Create compact transcript segment
        transcript_segment = ""
        for _, row in segment_df.iterrows():
            speaker = row['speaker'] if pd.notna(row['speaker']) else 'Unknown'
            text = row['Text'] if pd.notna(row['Text']) else ''
            transcript_segment += f"{speaker}: \"{text.strip()}\"\\n"
        
        # Create prompt
        prompt = create_compact_prompt(transcript_segment)
        
        segment_info = {
            'segment_id': i // utterances_per_segment + 1,
            'utterance_count': len(segment_df),
            'speakers_present': list(segment_df['speaker'].unique()),
            'transcript_segment': transcript_segment,
            'prompt': prompt
        }
        
        segments.append(segment_info)
    
    return segments

def segment_transcript_string_for_friction(values, utterances_per_segment=10):
    """Create transcript segments with compact prompts"""
    #filtered_df = df[df['speaker'].isin(['D1', 'D2', 'D3', 'Builder', 'Group'])].copy()
    segments = []
    
    for i in range(0, len(values), utterances_per_segment):
        segment = values[i:i+utterances_per_segment]
        speakers = []
        
        # Create compact transcript segment
        transcript_segment = ""
        for row in segment:
            x = row.split(":")
            if(len(x) == 2):
                speaker = x[0]
                speakers.append(speaker)
                transcript_segment += f"{row}\\n"
        
        # Create prompt
        prompt = create_compact_prompt(transcript_segment)
        
        segment_info = {
            'segment_id': i // utterances_per_segment + 1,
            'utterance_count': len(segment),
            'speakers_present': list(speakers),
            'transcript_segment': transcript_segment,
            'prompt': prompt
        }
        
        segments.append(segment_info)
    
    return segments

def generate_with_local_model(model, tokenizer, prompt, generation_args, device):
    """Generate response with local model"""
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.padding_side = 'right'
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].size(1) + generation_args["max_new_tokens"],
            temperature=generation_args["temperature"],
            top_p=generation_args["top_p"],
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    prompt_length = inputs['input_ids'].shape[-1]
    new_tokens = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text

def generate_with_openai(prompt, model="gpt-4o", max_tokens=1000, temperature=0.7):
    """Generate response with OpenAI API (updated for openai>=1.0.0)"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error: {str(e)}"

def load_local_model(model_path, base_model="llama3_8b_instruct"):
    """Load local model with LoRA adapter"""
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
        
        # Merge the model
        merged_model = lora_model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.padding_side = "right"
        
        return merged_model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None, None

def process_segments_with_multiple_models(segments, local_models, use_openai=True, generation_args=None, output_dir="friction_results"):
    """Process transcript segments with multiple models"""
    os.makedirs(output_dir, exist_ok=True)
    
    if generation_args is None:
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    all_results = {
        "segments_processed": len(segments),
        "models_used": [],
        "results": []
    }
    
    # Process with local models
    for model_name, model_path in local_models.items():
        logger.info(f"Processing with local model: {model_name}")
        
        model, tokenizer = load_local_model(model_path, "meta-llama/Meta-Llama-3-8B-Instruct")
        if model is None:
            continue
            
        all_results["models_used"].append(model_name)
        
        model_results = {
            "model_name": model_name,
            "model_type": "local",
            "segment_results": []
        }
        
        device = next(model.parameters()).device
        
        for segment in tqdm(segments, desc=f"Processing {model_name}"):
            try:
                generated_text = generate_with_local_model(
                    model, tokenizer, segment['prompt'], generation_args, device
                )
                
                # Parse the generated text
                parsed_components = parse_tags_robust(generated_text)
                
                segment_result = {
                    "segment_id": segment['segment_id'],
                    "speakers_present": segment['speakers_present'],
                    "utterance_count": segment['utterance_count'],
                    "full_prompt": segment['prompt'],
                    "generated_text": generated_text,
                    "parsed_components": {
                        "belief_state": ' '.join(parsed_components.get('belief_state', [])),
                        "common_ground": ' '.join(parsed_components.get('rationale', [])),
                        "friction": ' '.join(parsed_components.get('friction', []))
                    }
                }
                
                model_results["segment_results"].append(segment_result)
                
            except Exception as e:
                logger.error(f"Error processing segment {segment['segment_id']} with {model_name}: {str(e)}")
                continue
        
        all_results["results"].append(model_results)
        
        # Clean up model to free memory
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Process with OpenAI if requested
    if use_openai:
        logger.info("Processing with OpenAI GPT-4o")
        
        all_results["models_used"].append("gpt-4o")
        
        openai_results = {
            "model_name": "gpt-4o",
            "model_type": "openai",
            "segment_results": []
        }
        
        for segment in tqdm(segments, desc="Processing GPT-4o"):
            try:
                generated_text = generate_with_openai(
                    segment['prompt'],
                    model="gpt-4o",
                    max_tokens=generation_args["max_new_tokens"],
                    temperature=generation_args["temperature"]
                )
                
                # Parse the generated text
                parsed_components = parse_tags_robust(generated_text)
                
                segment_result = {
                    "segment_id": segment['segment_id'],
                    "speakers_present": segment['speakers_present'],
                    "utterance_count": segment['utterance_count'],
                    "full_prompt": segment['prompt'],
                    "generated_text": generated_text,
                    "parsed_components": {
                        "belief_state": ' '.join(parsed_components.get('belief_state', [])),
                        "common_ground": ' '.join(parsed_components.get('rationale', [])),
                        "friction": ' '.join(parsed_components.get('friction', []))
                    }
                }
                
                openai_results["segment_results"].append(segment_result)
                
                # Add small delay to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing segment {segment['segment_id']} with OpenAI: {str(e)}")
                continue
        
        all_results["results"].append(openai_results)
    
    # Save all results
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_file = f"{output_dir}/friction_analysis_results_{timestamp}.pkl"
    
    # with open(output_file, 'wb') as f:
    #     pickle.dump(all_results, f)
    
    # logger.info(f"Results saved to {output_file}")
    return all_results


def analyze_friction_results(results_file):
    """Analyze and summarize friction analysis results"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\\n=== FRICTION ANALYSIS SUMMARY ===")
    print(f"Segments processed: {results['segments_processed']}")
    print(f"Models used: {', '.join(results['models_used'])}")
    
    for model_result in results['results']:
        model_name = model_result['model_name']
        segment_count = len(model_result['segment_results'])
        
        print(f"\\n--- {model_name} Results ---")
        print(f"Segments completed: {segment_count}")
        
        # Sample output from first segment
        if model_result['segment_results']:
            first_segment = model_result['segment_results'][0]
            print(f"\\nSample output (Segment {first_segment['segment_id']}):")
            print(f"Speakers: {', '.join(first_segment['speakers_present'])}")
            print(f"Belief State: {first_segment['parsed_components']['belief_state'][:100]}...")
            print(f"Common Ground: {first_segment['parsed_components']['common_ground'][:100]}...")
            print(f"Friction: {first_segment['parsed_components']['friction'][:100]}...")

def start_server():
    HOST = '129.82.138.15'  # Standard loopback interface address (localhost)
    PORT = 65433       # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        # Model setup code from 
        # Set OpenAI API key (set your API key here or as environment variable)
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Define local models to test
        local_models = {
    #         "sft_deli_multiturn": 'sft_deli_multiturn_rogue_cleaned/checkpoint-3000',
    #         "deli_dpo": 'DELI_all_weights/DELI_dpo_weights/checkpoint-3500',
    #         "deli_sft": "DELI_all_weights/DELI_sft_weights/checkpoint-small-1500",
    #         "deli_ppo": "DELI_all_weights/DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800",
            "deli_faaf": '/home/traceteam/DELI_faaf/diplomacy_deli_weights/DELI_faaf_weights/checkpoint-2000',
        }
        
        # Generation arguments
        generation_args = {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # import glob
        # result_files = glob.glob("friction_analysis_results*/*.pkl")
        # if result_files:
        #     latest_results = max(result_files, key=os.path.getctime)
        #     analyze_friction_results(latest_results)
        # else:
        #     print("No result files found!")
        # return results

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        data = conn.recv(2048)
                        if not data:
                            break
                        print("Received Data Length:" + str(len(data)))
                        transcriptions = data.decode()
                        print(f"Transcriptions:\n{transcriptions}")
                        print("\nGenerating friction for dialogue...")

                        ################################ new friction processing
                        # Load transcript data
                        #df = pd.read_csv('group7_transcript.csv')
                        df = transcriptions.split('\n')
                        
                        # Create segments
                        segments = segment_transcript_string_for_friction(df, utterances_per_segment=20)
                        logger.info(f"Created {len(segments)} segments")

                        # Process segments with all models
                        results = process_segments_with_multiple_models(
                            segments=segments[:5],  # Process first 5 segments for testing
                            local_models=local_models,
                            use_openai=False,
                            generation_args=generation_args,
                            output_dir="friction_analysis_results_DPIP")

                        ################################

                        # TODO check results and fix parsing, this won't work since the results are now a dictionary
                        returnString = ''
                        if results is not None:
                            print(results)
                            conn.sendall(str.encode(returnString, 'utf-8')) 
                        else:
                            conn.sendall(str.encode("No Friction", 'utf-8'))
                        break
                    except ConnectionResetError as e:
                        print(f"Connection with {addr} was reset: {e}")
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        break

# def main():
#     """Main execution function"""
    
#     # Set OpenAI API key (set your API key here or as environment variable)
 
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
#     # Load transcript data
#     df = pd.read_csv('group7_transcript.csv')
    
#     # Create segments
#     segments = segment_transcript_for_friction(df, utterances_per_segment=20)
#     logger.info(f"Created {len(segments)} segments")
    
#     # Define local models to test
#     local_models = {
# #         "sft_deli_multiturn": 'sft_deli_multiturn_rogue_cleaned/checkpoint-3000',
# #         "deli_dpo": 'DELI_all_weights/DELI_dpo_weights/checkpoint-3500',
# #         "deli_sft": "DELI_all_weights/DELI_sft_weights/checkpoint-small-1500",
# #         "deli_ppo": "DELI_all_weights/DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800",
#         "deli_faaf": 'DELI_all_weights/DELI_faaf_weights/checkpoint-2000',
#     }
    
#     # Generation arguments
#     generation_args = {
#         "max_new_tokens": 400,
#         "temperature": 0.7,
#         "top_p": 0.9,
#     }
    
#     # Process segments with all models
#     results = process_segments_with_multiple_models(
#         segments=segments[:5],  # Process first 5 segments for testing
#         local_models=local_models,
#         use_openai=True,
#         generation_args=generation_args,
#         output_dir="friction_analysis_results_DPIP_group7"
#     )
    
#     import glob
#     result_files = glob.glob("friction_analysis_results*/*.pkl")
#     if result_files:
#         latest_results = max(result_files, key=os.path.getctime)
#         analyze_friction_results(latest_results)
#     else:
#         print("No result files found!")
#     return results

if __name__ == "__main__":
    start_server()
