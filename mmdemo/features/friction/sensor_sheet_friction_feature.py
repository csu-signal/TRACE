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
from mmdemo.interfaces import ColorImageInterface, DpipActionInterface, DpipCommonGroundTrackingInterface, DpipFrictionOutputInterface, FrictionOutputInterface, PropositionInterface, TranscriptionInterface
import tkinter as tk
from tkinter import Button, ttk
from PIL import ImageGrab
import os

from mmdemo.utils.files import create_tmp_dir_with_featureName

@final
class SensorSheetFrictionFeature(BaseFeature):
    """
    The friction feature for the sensor task
    """

    def __init__(
        self, transcription: BaseFeature[TranscriptionInterface]
    ):
        super().__init__(transcription)
        self.init = False
        self.useTabs = True
        #todo setup get thread for google api

    def initialize(self):
        print("Sensor Friction")

    def get_output(self, transcription: TranscriptionInterface):
        #if not transcription.is_new(): #TODO update to run only when transcription come in
        #    return None

        #TODO return friction for output
       return FrictionOutputInterface(
                    friction_statement="", transciption_subset=transcription)

    def worker(self):
        #print("New DPIP Interface Update Thread Started")
        # todo setup thread
        try:
            test = 0

        except Exception as e:
            print(f"SENSOR FRICTION FEATURE THREAD: An error occurred: {e}")
