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
    PlannerInterface,
    TranscriptionInterface,
)
from mmdemo.features.friction import friction_local

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
        plan: BaseFeature[PlannerInterface],
        *,
        host: str | None = None,
        port: int | None = 0,
        minUtteranceValue: int | None = 10,
    ):
        super().__init__(transcription, plan) 
        self.transcriptionHistory = []
        self.frictionSubset = []
        self.friction = ''
        self.subsetTranscriptions = ''
        self.t = threading.Thread(target=self.worker)
        self.minUtteranceValue = minUtteranceValue
        self.solvability_history = 0

        if host:
            self.HOST = host
        if port != 0:
            self.PORT = port
        self.LOCAL = False #Run local or remote #TODO

    def initialize(self):
        print("Friction Init HOST: " + str(self.HOST) + " PORT: " + str(self.PORT))
    
    def get_output(self, transcription: TranscriptionInterface, plan:PlannerInterface):
        if not transcription.is_new():
            return FrictionOutputInterface(friction_statement=self.friction, transciption_subset=self.subsetTranscriptions.replace("\n", " "))

        # with open("mmdemo/features/planner/benchmarks/problem.pddl", "r") as file:
        #     content = file.read()
        # match = re.search(r"\(:init\s*(.*?)\)\s*(?=\(:goal|\(:)", content, re.DOTALL)
        # if match:
        #     init_section = match.group(1).strip()

        # planner_output = plan.plan
        # compare_new_block = True
        # try:
        #     planner_step = [line for line in planner_output.split("\n") if "compare" in line][0]
        #     compare_blocks = re.findall(r"\b\w*block\w*\b", planner_step)
        #     # check if we are comparing a new block
        #     for block in compare_blocks:
        #         if block not in init_section:
        #             #if we need to compare a new block, move on
        #             compare_new_block = True
        #             break
        #         else:
        #             #if we need to look at existing blocks, intervene
        #             compare_new_block = False
        # except:
        #     pass
        # if not plan.solv or not compare_new_block:
        #     plan.solv = False
        if plan.solv:
            self.solvability_history = 0
        else:
            self.solvability_history += 1

        # transcription.text += "\nWe believe that " + ", ".join(plan.fbank) +"."

        #if the transcription text is empty don't add it to the history
        if transcription.text != '':
            self.transcriptionHistory.append(transcription.speaker_id + ": " + transcription.text)
            self.frictionSubset.append(transcription.speaker_id + ": " + transcription.text)
            # self.transcriptionHistory += "P1: " + transcription.text + "\n"
                    
        if not plan.solv and (self.solvability_history == self.minUtteranceValue or self.solvability_history == 1):
            self.solvability_history = 1
            if not self.t.is_alive() and not self.LOCAL:
                # do this process on the main thread so the socket thread doesn't miss any values
                # if there are less values in the friction subset the min utterance value pad the list with values from the history
                if(len(self.frictionSubset) < self.minUtteranceValue):
                    print(f'\nA minimum of {self.minUtteranceValue} utterances are needed to send to the friction LLM. There have been {len(self.frictionSubset)} utterance(s) since the last friction request. Attempting to add values from transcription history.')
                    if(len(self.transcriptionHistory) > self.minUtteranceValue):
                        self.frictionSubset = self.transcriptionHistory[-self.minUtteranceValue:]
                    else:
                        # if there are less values in the history then the min utterance value, use the full history
                        self.frictionSubset = self.transcriptionHistory

                #Add beliefs to 
                self.frictionSubset.append("\nWe believe that " + ", ".join(plan.fbank) +".")
                # format the transcriptions as a string to send over the socket
                self.subsetTranscriptions = ''
                for utter in self.frictionSubset:
                    self.subsetTranscriptions += utter + "\n"

                print("\nSubset of Transcriptions:\n" + self.subsetTranscriptions)
                self.t = threading.Thread(target=self.worker)
                self.t.start()
                self.frictionSubset = []
            elif self.LOCAL:
                #TODO
                friction_detector = friction_local.FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct") #this is the lora model id on huggingface (SFT model)
                #instead of calling FrictionInference as done above, add the generation arguments to specify parameters like max-length depending on what model you are calling
                    #for FAAF use, 356 and for SFT use 200 as shown below
                # define the generation args 
                custom_args_sft = {
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                        "do_sample": True,
                        "top_k": 50,
                        "top_p": 0.9
                    }

                custom_args_faaf = {
                        "max_new_tokens": 356,
                        "temperature": 0.9,
                        "do_sample": True,
                        "top_k": 50,
                        "top_p": 0.9
                    }
                # friction_detector = friction_local.FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct", generation_args = custom_args_sft) # instantiate only one of these friction_detector variable
                friction_detector = friction_local.FrictionInference("Abhijnan/intervention_agent", generation_args = custom_args_faaf)


                # friction_detector = friction_local.FrictionInference("Abhijnan/dpo_friction_run_with_69ksamples") #this is the dpo model
                friction_local.start_local(self.subsetTranscriptions,friction_detector)
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
                print(self.transcriptionHistory[-1])
        except Exception as e:
            self.friction = ''
            print(f"FRICTION FEATURE THREAD: An error occurred: {e}")

