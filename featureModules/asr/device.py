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
class AsrQueueData:
    id: str
    start: float
    stop: float
    audio_file: str


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
    def record_chunks(id, device_index, queue, done, chunk_length=5, rate=16000, chunk=1024, format=pyaudio.paInt16):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk, input_device_index=device_index)  # Use the selected device index
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
            queue.put(AsrQueueData(id, start, stop, next_file))

        stream.stop_stream()
        stream.close()
        p.terminate()

@final
class PrerecordedDevice(BaseDevice):
    READ_FRAME_COUNT = 1
    SAVE_INTERVAL_SECONDS = 5

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

    def create_writer(self, path):
        chunk_writer = wave.open(path, 'wb')
        chunk_writer.setnchannels(self.reader.getnchannels())
        chunk_writer.setsampwidth(self.reader.getsampwidth())
        chunk_writer.setframerate(self.reader.getframerate())
        return chunk_writer

    def save_if_needed(self):
        assert self.asr_queue is not None

        if self.audio_time - self.last_save_time >= self.SAVE_INTERVAL_SECONDS:
            next_file = fr"chunks\device{hash(self.path)}-{self.counter:05}.wav"
            chunk_writer = self.create_writer(next_file)
            chunk_writer.writeframes(self.frames)
            chunk_writer.close()

            self.last_save_time = self.audio_time
            self.frames = b''
            self.counter += 1

            self.asr_queue.put(AsrQueueData(self.get_id(), time.time() - self.SAVE_INTERVAL_SECONDS, time.time(), next_file))

    def handle_frame(self, frame_count: int):
        # while audio time < video time
        while self.audio_time < frame_count / self.video_frame_rate:
            self.frames += self.reader.readframes(self.READ_FRAME_COUNT)
            self.num_frames_read += self.READ_FRAME_COUNT

            self.save_if_needed()
