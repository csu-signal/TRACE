import multiprocessing as mp
import os
from ctypes import c_bool

import faster_whisper
import pyaudio

from featureModules.asr.device import AsrQueueData, BaseDevice
from featureModules.IFeature import *
from logger import Logger
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def select_audio_device():
    p = pyaudio.PyAudio()
    # create list of available devices
    print("Available devices:")
    for i in range(p.get_device_count()):
        print(i, ":", p.get_device_info_by_index(i).get('name'))
    # select device
    device_index = int(input("Select device index: "))
    print("Selected device:", p.get_device_info_by_index(device_index).get('name'))
    p.terminate()
    return device_index


def process_chunks(queue: "mp.Queue[AsrQueueData]", done, print_output=False, output_queue=None):
    # model = faster_whisper.WhisperModel("large-v2", compute_type="float16")
    model = faster_whisper.WhisperModel("small", compute_type="float16")
    while not done.value:
        data = queue.get()
        name, start, stop, chunk_file = data.id, data.start, data.stop, data.audio_file
        
        segments, info = model.transcribe(chunk_file, language="en")
        transcription = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5)  # Join segments into a single string

        if print_output:
            print(f'{name}: {transcription}')

        if output_queue is not None:
            output_queue.put((name, start, stop, transcription, chunk_file))


class AsrFeature(IFeature):
    def __init__(self, devices: list[BaseDevice], n_processors=1, csv_log_file=None):
        """
        devices should be of the form [(name, index), ...]
        """
        self.device_lookup = {d.get_id():d for d in devices}
        self.asr_output_queue = mp.Queue()

        asr_internal_queue = mp.Queue()
        done = mp.Value(c_bool, False)
        recorders = [d.create_recorder_process(asr_internal_queue, done) for d in devices]
        processors = [mp.Process(target=process_chunks, args=(asr_internal_queue, done), kwargs={"output_queue":self.asr_output_queue}) for _ in range(n_processors)]


        for i in recorders + processors:
            i.start()
        
        self.logger = Logger(file=csv_log_file)
        self.logger.write_csv_headers("frame", "name", "start", "stop", "text", "audio_file")


    def processFrame(self, frame, frame_count):
        utterances = []

        for id, device in self.device_lookup.items():
            device.handle_frame(frame_count)

        while not self.asr_output_queue.empty():
            id, start, stop, text, audio_file = self.asr_output_queue.get()
            if len(text.strip()) > 0:
                utterances.append((id, start, stop, text, audio_file))
                self.logger.append_csv(frame_count, name, start, stop, text, audio_file)


        cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return utterances
