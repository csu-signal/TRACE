from featureModules.IFeature import *
from utils import *
import multiprocessing as mp
from featureModules.asr.full_time_recording import record_chunks, process_chunks
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


    def processFrame(self, deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus):
        transcriptions = {n:"" for n in self.full_transcriptions.keys()}
        while not self.asr_output_queue.empty():
            name, s = self.asr_output_queue.get()
            self.full_transcriptions[name] += s
            transcriptions[name] += s

        for name,s in transcriptions.items():
            if len(s.strip()) > 0:
                print(f"{name}: {s}")

        cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
