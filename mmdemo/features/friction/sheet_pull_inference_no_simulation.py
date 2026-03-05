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
from model_configs import load_local_model, load_local_base_and_lora_model #import the local model loading logic 
import sheet_pull_inference
from sheet_pull_inference import (
    run_simulation, clear_demo_sheet, execute_with_retry
) 
import socket
import threading

CREDENTIALS_PATH = "/credentials.json" #get google sheet api credentials, the OAuth app identity, downloaded once from Google Cloud Console. Identifies your app to Google. Doesn't change unless you regenerate it. Contains client_id, client_secret, redirect URIs. 
TOKEN_PATH = "token.pickle" # the user's access/refresh token, generated at runtime after the user completes the OAuth flow (Visit this URL → paste code).
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

REAL_SHEET_ID = os.getenv("REAL_SHEET_ID") #google sheet shared from ISAT
DEMO_SHEET_ID = os.getenv("DEMO_SHEET_ID") #create a demo blank sheet with an ID, 

GROUP = ['412', '413', '417']

# Store transcript received from separate connection
received_transcript = None  

TRANSCRIPT_PORT = 65434
HOST = '129.82.138.15'

def start_transcript_listener():
    global received_transcript
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, TRANSCRIPT_PORT))
        s.listen()
        print(f"Transcript listener on {HOST}:{TRANSCRIPT_PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    received_transcript = data.decode('utf-8')
                    print(f"Transcript updated ({len(received_transcript)} chars)")
                    conn.sendall(b"ack")
                except Exception as e:
                    print(f"Transcript listener error: {e}")

def build_intervention_prompt_with_transcript(deltas, current_state, current_rows, group):
    # Get base prompt from original function
    base_prompt = sheet_pull_inference.build_intervention_prompt(deltas, current_state, current_rows, group)
    
    # Inject transcript before the final output instruction
    if received_transcript:
        transcript_block = f"\nRecent spoken transcript:\n{received_transcript}\n"
        # Insert before the output format instruction
        base_prompt = base_prompt.replace(
            "Based on the recent updates",
            f"{transcript_block}\nBased on the recent updates"
        )
    return base_prompt


def start_server(sheets_service, merged_model, tokenizer, model_name, generation_args):
    HOST = '129.82.138.15'
    PORT = 65433
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    # Get real sheet rows
                    result = execute_with_retry(
                        sheets_service.spreadsheets().values().get(
                            spreadsheetId=REAL_SHEET_ID, range="A1:Z50"
                        )
                    )
                    rows = result.get('values', [])

                    # Clear + write headers to demo sheet
                    clear_demo_sheet(sheets_service, DEMO_SHEET_ID)
                    execute_with_retry(
                        sheets_service.spreadsheets().values().update(
                            spreadsheetId=DEMO_SHEET_ID,
                            range="A1", valueInputOption='RAW',
                            body={'values': rows[:2]}
                        )
                    )

                    # Run full simulation
                    run_simulation(
                        rows, GROUP, DEMO_SHEET_ID,
                        sheets_service, merged_model, tokenizer, model_name, generation_args,
                        poll_interval=5, delay_range=(8, 15), prompt_fn=build_intervention_prompt_with_transcript 
                    )

           
                    conn.sendall(str.encode("simulation_complete"))
                except Exception as e:
                    print(f"Error: {e}")
                    conn.sendall(str.encode("error"))

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
    base_llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_models = {
#         "sft_deli_multiturn": 'sft_deli_multiturn_rogue_cleaned/checkpoint-3000',
#         "deli_dpo": 'DELI_all_weights/DELI_dpo_weights/checkpoint-3500',
#         "deli_sft": "DELI_all_weights/DELI_sft_weights/checkpoint-small-1500",
#         "deli_ppo": "DELI_all_weights/DELI_ppo_weights/ppo_checkpoint_epoch_1_batch_800",
        "deli_faaf": '/home/traceteam/DELI_faaf/diplomacy_deli_weights/DELI_faaf_weights/checkpoint-2000',
        #"wtd_faaf_new": '/home/traceteam/DELI_faaf/diplomacy_deli_weights/DELI_faaf_weights/checkpoint-3000/checkpoint-3000'
    }
    model_path = local_models['deli_faaf']

    generation_args = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
}

    merged_model, tokenizer = load_local_model(model_path, base_model = base_llama_path)
    model_name = model_path.split("/")[-2]

    #listen to the transcript coming if (probably?)
    threading.Thread(target=start_transcript_listener, daemon=True).start()
    start_server(sheets_service, merged_model, tokenizer, model_name, generation_args)

    #rough dummpy workflow with transcripts coming in 
    # t=0   received_transcript = None
    # t=30  transcript arrives → received_transcript = "Student 412: ..."
    # t=35  delta detected → prompt built WITH transcript injected
    # t=60  another delta → still uses same transcript (no new one came in)
    # t=90  new transcript arrives → received_transcript = "Student 413: ..."
    # t=95  delta detected → prompt built with NEW transcript
