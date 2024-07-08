from collections import defaultdict
from featureModules.IFeature import *
from utils import *
import multiprocessing as mp
from featureModules.asr.live_transcription import record_chunks, process_chunks
from ctypes import c_bool
import os

class AsrFeature(IFeature):
    def __init__(self, devices: list[tuple[str, int]], n_processors=1):
        """
        devices should be of the form [(name, index), ...]
        """
        self.asr_output_queue = mp.Queue()
        self.full_transcriptions = {n:"" for n,i in devices}

        asr_internal_queue = mp.Queue()
        done = mp.Value(c_bool, False)
        recorders = [mp.Process(target=record_chunks, args=(name, index, asr_internal_queue, done)) for name,index in devices]
        processors = [mp.Process(target=process_chunks, args=(asr_internal_queue, done), kwargs={"output_queue":self.asr_output_queue}) for _ in range(n_processors)]

        os.makedirs("chunks", exist_ok=True)

        for i in recorders + processors:
            i.start()


    def processFrame(self, frame):
        utterances = []

        while not self.asr_output_queue.empty():
            name, start, stop, text = self.asr_output_queue.get()
            self.full_transcriptions[name] += text
            if len(text.strip()) > 0:
                utterances.append((name, start, stop, text))
                # with open("asr_out.csv", "a") as f:
                #     f.write(f"{name},{start},{stop},\"{text}\"\n")

        cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return utterances
