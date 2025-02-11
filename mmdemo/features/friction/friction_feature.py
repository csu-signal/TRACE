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
        port: int | None = 0,
        minUtteranceValue: int | None = 10
    ):
        super().__init__(transcription) 
        self.transcriptionHistory = []
        self.frictionSubset = []
        self.friction = ''
        self.subsetTranscriptions = ''
        self.t = threading.Thread(target=self.worker)
        self.minUtteranceValue = minUtteranceValue

        if host:
            self.HOST = host
        if port != 0:
            self.PORT = port

    def initialize(self):
        print("Friction Init HOST: " + str(self.HOST) + " PORT: " + str(self.PORT))
    
    def get_output(self, transcription: TranscriptionInterface):
        if not transcription.is_new():
            return FrictionOutputInterface(friction_statement=self.friction, transciption_subset=self.subsetTranscriptions.replace("\n", " "))

        #if the transcription text is empty don't add it to the history
        if transcription.text != '':
            self.transcriptionHistory.append(transcription.speaker_id + ": " + transcription.text)
            self.frictionSubset.append(transcription.speaker_id + ": " + transcription.text)
            # self.transcriptionHistory += "P1: " + transcription.text + "\n"

        if not self.t.is_alive():
            # do this process on the main thread so the socket thread doesn't miss any values
            # if there are less values in the friction subset the min utterance value pad the list with values from the history
            if(len(self.frictionSubset) < self.minUtteranceValue):
                print(f'\nA minimum of {self.minUtteranceValue} utterances are needed to send to the friction LLM. There have been {len(self.frictionSubset)} utterance(s) since the last friction request. Attempting to add values from transcription history.')
                if(len(self.transcriptionHistory) > self.minUtteranceValue):
                    self.frictionSubset = self.transcriptionHistory[-self.minUtteranceValue:]
                else:
                    # if there are less values in the history then the min utterance value, use the full history
                    self.frictionSubset = self.transcriptionHistory

            # format the transcriptions as a string to send over the socket
            self.subsetTranscriptions = ''
            for utter in self.frictionSubset:
                self.subsetTranscriptions += utter + "\n"

            print("\nSubset of Transcriptions:\n" + self.subsetTranscriptions)
            self.t = threading.Thread(target=self.worker)
            self.t.start()
            self.frictionSubset = []
        else:
            print("Friction request in progress...waiting for the thread to complete")

        return FrictionOutputInterface(
                friction_statement=self.friction, transciption_subset=self.subsetTranscriptions.replace("\n", " "))
    
    def worker(self):
        print("New Friction Request Thread Started")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.HOST, self.PORT))
                sendData = str.encode(self.subsetTranscriptions)
                print("Send Data Length:" + str(len(sendData))) 
                s.sendall(sendData)
                print("Waiting for friction server response")
                data = s.recv(2048)
            received = data.decode()
            if received != "No Friction":
                self.friction = received
                print(f"Received from Server:{received}")
        except Exception as e:
            self.friction = ''
            print(f"FRICTION FEATURE THREAD: An error occurred: {e}")

