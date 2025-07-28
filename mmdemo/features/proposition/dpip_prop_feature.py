import csv
from pathlib import Path
import pickle
import socket
import threading
from typing import final
import copy

import nltk
from sentence_transformers import SentenceTransformer

from mmdemo.base_feature import BaseFeature
from mmdemo.features.friction import friction_local
from mmdemo.features.proposition.demo import load_model, process_sentence
from mmdemo.features.proposition.demo_helpers import get_pickle
from mmdemo.interfaces import DpipActionInterface, DpipFrictionOutputInterface, DpipObjectInterface3D, TranscriptionInterface


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
        objects: BaseFeature[DpipObjectInterface3D],
        actions: BaseFeature[DpipActionInterface],
        #plan: BaseFeature[PlannerInterface], commented out for now
        *,
        host: str | None = None,
        port: int | None = 0,
        minUtteranceValue: int | None = 10,
        csvSupport: str | None = None
    ):
        super().__init__(transcription, objects, actions) 
        self.transcriptionHistory = {}
        self.transcriptionIndex = 0
        self.frictionSubset = {}
        self.friction = ''
        self.cg = 'None'
        self.subsetTranscriptions = ''
        self.t = threading.Thread(target=self.worker)
        self.minUtteranceValue = minUtteranceValue
        self.solvability_history = 0
        self.csvSupport = csvSupport
        self.lastUtterance = 0
        self.currentStructure = {}
        self.startingIndex = 0
        self.endingIndex = 0
        self.timestamp = 0

        if host:
            self.HOST = host
        if port != 0:
            self.PORT = port
        self.LOCAL = False #Run local or remote #TODO

    def initialize(self):
        print("DPIP LLM Friction Init HOST: " + str(self.HOST) + " PORT: " + str(self.PORT))

    def get_output(self, transcription: TranscriptionInterface, objects: DpipObjectInterface3D, actions: DpipActionInterface):
        if not transcription.is_new() or not actions.is_new():
            return DpipFrictionOutputInterface(friction_statement=self.friction, cg_json=self.cg, transciption_subset=self.subsetTranscriptions.replace("\n", " "))

        # if plan.solv:
        #     self.solvability_history = 0
        # else:
        #     self.solvability_history += 1

        # transcription.text += "\nWe believe that " + ", ".join(plan.fbank) +"."
        self.currentStructure[self.transcriptionIndex] = copy.deepcopy(actions.structure)
        self.timestamp = objects.frame_index

        #if the transcription text is empty don't add it to the history
        if transcription.text != '':
            t = "\"" + transcription.text.strip('"').strip() + "\""
            if self.csvSupport != None:
                csv_file = csv.reader(open(self.csvSupport, "r"), delimiter=",")
                for row in csv_file:
                    if transcription.text != '' and row[4] != '' and row[4].strip('"').strip() in transcription.text.strip('"').strip() and self.lastUtterance < float(row[0]):
                        if row[3] != '':
                            self.lastUtterance = float(row[0])
                            transcription.speaker_id = row[3]
                            print (row)
                        break

            if transcription.speaker_id != "Group" and transcription.speaker_id != "Instructor":
                self.transcriptionHistory[self.transcriptionIndex] = (transcription.speaker_id + ": " + t)
                self.frictionSubset[self.transcriptionIndex] = (transcription.speaker_id + ": " + t)
                self.transcriptionIndex = self.transcriptionIndex + 1
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
                        self.frictionSubset = dict(list(self.transcriptionHistory.items())[-self.minUtteranceValue:])
                    else:
                        # if there are less values in the history then the min utterance value, use the full history
                        self.frictionSubset = self.transcriptionHistory

                #Add beliefs to 
                # self.frictionSubset.append("\nWe believe that " + ", ".join(plan.fbank) +".")
                # format the transcriptions as a string to send over the socket
                self.subsetTranscriptions = ''
                for utter in self.frictionSubset:
                    self.subsetTranscriptions += self.frictionSubset[utter] + "\n"

                print("\nSubset of Transcriptions:\n" + self.subsetTranscriptions)
                if len(self.frictionSubset) >= 10:
                    self.startingIndex = list(self.frictionSubset)[0]
                    self.endingIndex = list(self.frictionSubset)[-1]
                    self.t = threading.Thread(target=self.worker)
                    self.t.start()
                    self.frictionSubset = {}
                else:
                     print("A minimum of 10 utterances are required to make a request.")
            else:
                print("Friction request in progress...waiting for the thread to complete")

            return DpipFrictionOutputInterface(
                    friction_statement=self.friction, cg_json=self.cg, transciption_subset=self.subsetTranscriptions.replace("\n", " "))
    
    def worker(self):
        print("New DPIP Friction Request Thread Started")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.HOST, self.PORT))
                my_object = {"transcripts": self.subsetTranscriptions, "start_structure": self.currentStructure[self.startingIndex],  "end_structure": self.currentStructure[self.endingIndex], "timestamp":self.timestamp}
                serialized_data = pickle.dumps(my_object)
                print("Send Data Length:" + str(len(serialized_data))) 
                s.sendall(serialized_data)
                
                print("Waiting for friction server response")
                data = s.recv(2048)
            deserialized_object = pickle.loads(data)
            cg = deserialized_object["commonGround"]
            friction = deserialized_object["friction"]
            if cg != '':
                self.cg = cg
            if friction != '':
                self.friction = friction
            print(f"Received from Server:{deserialized_object}")
        except Exception as e:
            self.friction = ''
            print(f"DPIP PROP FEATURE THREAD: An error occurred: {e}")
