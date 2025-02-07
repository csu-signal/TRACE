import socket
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
from mmdemo.features.gesture.helpers import get_average_hand_pixel, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    FrictionOutputInterface,
    TranscriptionInterface,
)

@final
class Friction(BaseFeature[FrictionOutputInterface]):
    """
    Detect friction in group work (client side).

    Input interfaces are `TranscriptionInterface`, ...

    Output interface is `FrictionOutputInterface`

    Keyword arguments:
    `host` -- the host of the friction server
    `port` -- the server port
    """

    HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
    PORT = 65432  # The port used by the server 

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        *,
        host: str | None = None,
        port: int | None = 0
    ):
        super().__init__(transcription) 
        self.transcriptionHistory = ''
        self.friction = ''
        self.t = threading.Thread(target=self.worker)

        if host:
            self.HOST = host
        if port != 0:
            self.PORT = port

    def initialize(self):
        print("Friction Init HOST: " + str(self.HOST) + " PORT: " + str(self.PORT))
    
    def get_output(self, transcription: TranscriptionInterface):
        if not transcription.is_new():
            return FrictionOutputInterface(friction_statement=self.friction)

        #if the transcription text is empty don't add it to the history
        if transcription.text != '':
            self.transcriptionHistory += transcription.speaker_id + ": " + transcription.text + "\n"
            # self.transcriptionHistory += "P1: " + transcription.text + "\n"
        print("Transcription History:\n" + self.transcriptionHistory)

        if not self.t.is_alive():
            self.t = threading.Thread(target=self.worker)
            self.t.start()
        else:
            print("Friction thread is already running...waiting for the next frame")

        return FrictionOutputInterface(
                friction_statement=self.friction)
    
    def worker(self):
        print("Friction Thread Started")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.HOST, self.PORT))
                sendData = str.encode(self.transcriptionHistory)
                print("Send Data Length:" + str(len(sendData)))
                s.sendall(sendData)
                data = s.recv(2048)
            received = data.decode("utf-8")
            if received != "No Friction":
                self.friction = received
                print(f"Received from Server:{received}")
        except Exception as e:
            self.friction = ''
            print(f"FRICTION FEATURE: An error occurred: {e}")

