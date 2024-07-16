import multiprocessing as mp
import os
import time
import wave
from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Synchronized
from typing import final

import pyaudio


class BaseDevice(ABC):
    @abstractmethod
    def get_id(self):
        raise NotImplementedError

    @abstractmethod
    def create_recorder_process(self, asr_queue: mp.Queue, done: Synchronized):
        raise NotImplementedError

    def handle_frame(self, frame_count: int):
        pass

@final
class MicDevice(BaseDevice):
    def __init__(self, name, mic_index) -> None:
        super().__init__()
        self.name = name
        self.index = mic_index

    def get_id(self):
        return self.name

    def create_recorder_process(self, asr_queue: mp.Queue, done: Synchronized):
        os.makedirs("chunks", exist_ok=True)
        return mp.Process(target=MicDevice.record_chunks, args=(self.name, self.index, asr_queue, done))

    @staticmethod
    def record_chunks(device_name, device_index, queue, done, chunk_length=5, rate=16000, chunk=1024, format=pyaudio.paInt16):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=device_index)  # Use the selected device index
        counter = 0
        while not done.value:
            next_file = fr"chunks\device{device_index}-{counter:05}.wav"
            counter += 1
            frames = []
            start = time.time()
            for i in range(0, int(rate / chunk * chunk_length)):
                data = stream.read(chunk)
                frames.append(data)
            stop = time.time()
            wf = wave.open(next_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            queue.put((device_name, start, stop, next_file))

        stream.stop_stream()
        stream.close()
        p.terminate()

@final
class PrerecordedDevice(BaseDevice):
    def __init__(self, path, video_frame_rate):
        super().__init__()
        self.path = path
        self.video_frame_rate = video_frame_rate
