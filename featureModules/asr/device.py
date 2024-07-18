import multiprocessing as mp
import os
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized
from typing import final

import numpy as np
import pyaudio


@dataclass
class AsrDeviceData:
    id: str
    start: float
    stop: float
    frames: bytes
    sample_rate: int
    sample_width: int
    channels: int


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
        return mp.Process(target=MicDevice.record_chunks, args=(self.get_id(), self.index, asr_queue, done))

    @staticmethod
    def record_chunks(id, device_index, queue, done, chunk_length_seconds=0.5, rate=16000, frames_per_read=512, format=pyaudio.paInt16):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=frames_per_read, input_device_index=device_index)  # Use the selected device index
        while not done.value:
            frames = []
            start = time.time()
            for i in range(0, int(rate / frames_per_read * chunk_length_seconds)):
                data = stream.read(frames_per_read)
                frames.append(data)
            stop = time.time()
            queue.put(AsrDeviceData(
                id,
                start,
                stop,
                b''.join(frames),
                rate,
                p.get_sample_size(format),
                1
            ))

        stream.stop_stream()
        stream.close()
        p.terminate()

@final
class PrerecordedDevice(BaseDevice):
    READ_FRAME_COUNT = 512
    SAVE_INTERVAL_SECONDS = 0.5

    def __init__(self, name, path, video_frame_rate=30):
        super().__init__()
        self.name = name
        self.path = path
        self.video_frame_rate = video_frame_rate
        self.asr_queue = None

        self.reader = wave.open(self.path, 'rb')
        self.num_frames_read = 0
        self.last_save_time = 0
        self.counter = 0

        self.frames = b''

    def get_id(self):
        return self.name

    def create_recorder_process(self, asr_queue: mp.Queue, done: Synchronized):
        os.makedirs("chunks", exist_ok=True)
        self.asr_queue = asr_queue
        return mp.Process(target=PrerecordedDevice.noop)

    @staticmethod
    def noop():
        pass

    @property
    def audio_time(self):
        return self.num_frames_read / self.reader.getframerate()

    def save_if_needed(self):
        assert self.asr_queue is not None

        if self.audio_time - self.last_save_time >= self.SAVE_INTERVAL_SECONDS:
            self.asr_queue.put(AsrDeviceData(
                self.get_id(),
                time.time() - self.SAVE_INTERVAL_SECONDS,
                time.time(),
                self.frames,
                self.reader.getframerate(),
                self.reader.getsampwidth(),
                self.reader.getnchannels()
            ))

            self.last_save_time = self.audio_time
            self.frames = b''
            self.counter += 1


    def handle_frame(self, frame_count: int):
        # while audio time < video time
        while self.audio_time < frame_count / self.video_frame_rate:
            self.frames += self.reader.readframes(self.READ_FRAME_COUNT)
            self.num_frames_read += self.READ_FRAME_COUNT

            self.save_if_needed()
