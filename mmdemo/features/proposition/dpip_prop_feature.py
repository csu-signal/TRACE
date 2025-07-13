from pathlib import Path
import socket
import threading
from typing import final

import nltk
from sentence_transformers import SentenceTransformer

from mmdemo.base_feature import BaseFeature
from mmdemo.features.friction import friction_local
from mmdemo.features.proposition.demo import load_model, process_sentence
from mmdemo.features.proposition.demo_helpers import get_pickle
from mmdemo.interfaces import DpipFrictionOutputInterface, TranscriptionInterface


@final
class DpipProposition(BaseFeature[DpipFrictionOutputInterface]):
    """
    Extract propositions from a transcription.

    Input interface is `TranscriptionInterface`

    Output interface is `DpipFrictionOutputInterface`

    Keyword arguments:
    `model_path` -- the path to the model (or None to use the default)
    """

    HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
    PORT = 65433  # The port used by the server 

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        #plan: BaseFeature[PlannerInterface], commented out for now
        *,
        host: str | None = None,
        port: int | None = 0,
        minUtteranceValue: int | None = 10,
    ):
        super().__init__(transcription) 
        self.transcriptionHistory = []
        self.frictionSubset = []
        self.friction = ''
        self.cg = 'None'
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
        print("DPIP LLM Friction Init HOST: " + str(self.HOST) + " PORT: " + str(self.PORT))

    def get_output(self, transcription: TranscriptionInterface):
        if not transcription.is_new():
            return DpipFrictionOutputInterface(friction_statement=self.friction, cg_json=self.cg, transciption_subset=self.subsetTranscriptions.replace("\n", " "))

        # if plan.solv:
        #     self.solvability_history = 0
        # else:
        #     self.solvability_history += 1

        # transcription.text += "\nWe believe that " + ", ".join(plan.fbank) +"."

        #if the transcription text is empty don't add it to the history
        if transcription.text != '':
            t = "\"" + transcription.text.strip('"').strip() + "\""
            self.transcriptionHistory.append(transcription.speaker_id + ": " + t)
            self.frictionSubset.append(transcription.speaker_id + ": " + t)
            # self.transcriptionHistory += "P1: " + transcription.text + "\n"
                    
        #if not plan.solv and (self.solvability_history == self.minUtteranceValue or self.solvability_history == 1):
        if True:
            self.solvability_history = 1
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

                #Add beliefs to 
                # self.frictionSubset.append("\nWe believe that " + ", ".join(plan.fbank) +".")
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

            #TODO include props and board state info for the GUI
            return DpipFrictionOutputInterface(
                    friction_statement=self.friction, cg_json=self.cg, transciption_subset=self.subsetTranscriptions.replace("\n", " "))
    
    def worker(self):
        print("New DPIP Friction Request Thread Started")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.HOST, self.PORT))
                sendData = str.encode(self.subsetTranscriptions)
                print("Send Data Length:" + str(len(sendData))) 
                s.sendall(sendData)
                print("Waiting for friction server response")
                data = s.recv(2048)
            received = data.decode()
            if received != " ":
                #self.friction = received #TODO include and parse friction statement
                self.cg = received #for now, will eventually parse
                print(f"Received from Server:{received}")
                print(self.transcriptionHistory[-1])
        except Exception as e:
            self.friction = ''
            print(f"DPIP PROP FEATURE THREAD: An error occurred: {e}")
