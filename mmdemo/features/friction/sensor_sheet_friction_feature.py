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
from mmdemo.features.friction.sensor_friction_helpers import run_inference_socket, load_local_model, get_sheets_service, poll_and_diff, build_intervention_prompt, run_inference, load_model
import tkinter as tk
from tkinter import Button, ttk
from PIL import ImageGrab
import os

from mmdemo.utils.files import create_tmp_dir_with_featureName

GROUP_IDS = ["412", "413", "417"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.pickle"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = "1P1lnXvxb3KXQym6qCtq2GdALJjaPPeCY8R9IsTcra9I"

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
        self.latest_friction = ""
        self.spreadsheet_id = SPREADSHEET_ID
        self.group_ids = GROUP_IDS
        self.t = threading.Thread(target=self.worker)


    def initialize(self):
        self.sheets_service = get_sheets_service()
        self.previous_state = {}
        
        # Start the polling worker thread once, as a daemon
        self.t = threading.Thread(target=self.worker, daemon=True)
        self.t.start()
        print("[SensorSheetFrictionFeature] Worker thread started")


    def get_output(self, transcription: TranscriptionInterface):
        # Always return the latest friction statement
        # The worker thread updates self.latest_friction continuously in the background
        return SensorSheetFrictionOutputInterface(friction_statement=self.latest_friction)

    def worker(self):
        try:
            print("[SensorSheetFrictionFeature.worker] Started polling loop")
            while True:
                deltas, current_state, current_rows = poll_and_diff(
                    self.sheets_service,
                    self.spreadsheet_id,
                    self.previous_state,
                    self.group_ids,
                )

                if deltas:
                    print(f"[SensorSheetFrictionFeature.worker] Detected {len(deltas)} deltas, calling LLM...")
                    prompt = build_intervention_prompt(deltas, current_state, current_rows, GROUP_IDS)
                    output = run_inference_socket(prompt)
                    print(f"[SensorSheetFrictionFeature.worker] LLM response received: {output if output else 'None'}...")
                    self.latest_friction = output
                    self.previous_state = current_state
                else:
                    print(f"[SensorSheetFrictionFeature.worker] No deltas at {time.time()}")

                time.sleep(5)

        except Exception as e:
            print(f"[SensorSheetFrictionFeature.worker] ERROR: {e}")
            import traceback
            traceback.print_exc()
