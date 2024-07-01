from featureModules.IFeature import *
from utils import *
import multiprocessing as mp
from featureModules.asr.full_time_recording import record_chunks, process_chunks
from ctypes import c_bool

# full_time_recording.select_audio_device
AUDIO_DEVICE_INDEX = 1

class AsrFeature(IFeature):
    def __init__(self):
        self.asr_output_queue = mp.Queue()
        self.full_transcription = ""

        asr_internal_queue = mp.Queue()
        done = mp.Value(c_bool, False)
        recorder = mp.Process(target=record_chunks, args=(AUDIO_DEVICE_INDEX, asr_internal_queue, done))
        processor = mp.Process(target=process_chunks, args=(AUDIO_DEVICE_INDEX, asr_internal_queue, done), kwargs={"output_queue":self.asr_output_queue})

        recorder.start()
        processor.start()

    def processFrame(self, deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus):
        while not self.asr_output_queue.empty():
            self.full_transcription += self.asr_output_queue.get()
        print(self.full_transcription)
        cv2.putText(frame, "ASR running", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
