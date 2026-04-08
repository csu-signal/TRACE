import json
import shutil
import socket
import time
import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np
import torch
import re
from typing import Dict, List, Optional
import threading
from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, DpipActionInterface, DpipCommonGroundTrackingInterface, SensorSheetFrictionOutputInterface, FrictionOutputInterface, PropositionInterface, TranscriptionInterface
from mmdemo.features.friction.sensor_friction_helpers import run_inference_socket, load_local_model, get_sheets_service, poll_and_diff, build_intervention_prompt, run_inference, load_model, update_recent_transcriptions
import tkinter as tk
from tkinter import Button, ttk
from PIL import ImageGrab
import os

from mmdemo.utils.files import create_tmp_dir_with_featureName

GROUP_IDS = ["412", "413", "417"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.pickle"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = "11TzA0If5M0iOuUw-vk1NvmnFWbSRG_9K6Qi_YWv04SE"

@final
class SensorSheetFrictionFeature(BaseFeature[SensorSheetFrictionOutputInterface]):
    """
    The friction feature for the sensor task
    """

    def __init__(
        self, transcription: BaseFeature[TranscriptionInterface]
    ):
        super().__init__(transcription)
        # self.init = False
        # self.useTabs = True
        self.history_transcriptions = []
        self.latest_friction = ""
        self.spreadsheet_id = SPREADSHEET_ID
        self.group_ids = GROUP_IDS
        self.t = threading.Thread(target=self.worker)
        self.llm_io_records = []
        self.llm_io_path = None


    def initialize(self):
        self.sheets_service = get_sheets_service()
        self.previous_state = {}
        repo_root = Path(__file__).resolve().parents[3]
        log_dir = repo_root / "logging-output-sensor"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.llm_io_path = log_dir / "sensor_llm_io.json"
        
        # Start the polling worker thread once, as a daemon
        self.t = threading.Thread(target=self.worker, daemon=True)
        self.t.start()


    def _write_llm_io_log(self):
        if self.llm_io_path is None:
            return

        with self.llm_io_path.open("w", encoding="utf-8") as log_file:
            json.dump(self.llm_io_records, log_file, indent=2)


    def _record_llm_io(self, prompt: str, output: str, deltas, response_time_seconds=None):
        self.llm_io_records.append(
            {
                "prompt": prompt,
                "output": output,
                "delta_count": len(deltas),
                "len_prompt": len(prompt),
                "response_time_seconds": response_time_seconds,
            }
        )
        self._write_llm_io_log()

    def _extract_group_friction_from_json(self, json_string: str) -> str:
        """
        Extract the group-level friction text from the LLM's JSON response.
        Expected format: {"group": {"text": "...", "reasoning": "..."}, ...}
        Returns the group's text field, or an empty string if parsing fails.
        """
        try:
            data = json.loads(json_string)
            if isinstance(data, dict) and "group" in data:
                group_data = data["group"]
                if isinstance(group_data, dict) and "text" in group_data:
                    text = group_data["text"]
                    if isinstance(text, str):
                        return text.strip()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[SensorSheetFrictionFeature] Warning: Failed to parse JSON response: {e}")
        
        # Return empty string if extraction fails
        return ""


    def get_output(self, transcription: TranscriptionInterface):
        if transcription and transcription.is_new() and transcription.text.strip():
            self.history_transcriptions = update_recent_transcriptions(
                self.history_transcriptions,
                transcription,
            )
        # Always return the latest friction statement
        # The worker thread updates self.latest_friction continuously in the background
        return SensorSheetFrictionOutputInterface(friction_statement=self.latest_friction)

    def worker(self):
        try:
            while True:
                deltas, current_state, current_rows = poll_and_diff(
                    self.sheets_service,
                    self.spreadsheet_id,
                    self.previous_state,
                    self.group_ids,
                )

                # Always advance baseline state so deletions/empty cells are tracked
                # even when they do not create deltas.
                self.previous_state = current_state

                if deltas:
                    print(f"[Sheet] Detected {len(deltas)} new change(s)")
                    prompt = build_intervention_prompt(
                        deltas,
                        current_state,
                        current_rows,
                        GROUP_IDS,
                        self.history_transcriptions,
                    )
                    output = ""
                    response_time_seconds = None
                    try:
                        response_start = time.perf_counter()
                        output = run_inference_socket(prompt)
                        response_time_seconds = time.perf_counter() - response_start
                        print("\n" + "=" * 24 + " LLM FULL RESPONSE " + "=" * 24)
                        print(output)
                    except Exception as e:
                        print(f"[SensorSheetFrictionFeature.worker] ERROR: {e}")
                    print("=" * 67 + "\n")
                    self._record_llm_io(prompt, output, deltas, response_time_seconds)
                    # Extract group-level friction from JSON response
                    self.latest_friction = self._extract_group_friction_from_json(output)

                time.sleep(5)

        except Exception as e:
            print(f"[SensorSheetFrictionFeature.worker] ERROR: {e}")
            import traceback
            traceback.print_exc()
