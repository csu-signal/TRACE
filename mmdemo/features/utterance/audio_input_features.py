import hashlib
import multiprocessing as mp
import os
import time
import wave
from collections import deque
from ctypes import c_bool
from pathlib import Path
from typing import final

import pyaudio

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import AudioFileInterface, ColorImageInterface


# TODO: if updates are happening slower than audio, we might need to either accumulate data to send in one big file or return a new interface which has a list of files
@final
class MicAudio(BaseFeature[AudioFileInterface]):
    def __init__(self, *, device_id: int, speaker_id: str | None = None) -> None:
        super().__init__()
        self.device_id = device_id
        self.speaker_id = speaker_id if speaker_id is not None else f"mic{device_id:03}"
        self.output_dir = Path("chunks") / f"mic{device_id:03}"

    def initialize(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.queue = mp.Queue()
        self.done = mp.Value(c_bool, False)

        self.process = mp.Process(
            target=MicAudio.record_chunks,
            args=(
                self.speaker_id,
                self.device_id,
                self.queue,
                self.done,
                self.output_dir,
            ),
        )
        self.process.start()

    def finalize(self):
        self.done.value = True
        self.process.join()

    def get_output(self) -> AudioFileInterface | None:
        if self.queue.empty():
            return None
        return self.queue.get()

    @staticmethod
    def record_chunks(
        speaker_id,
        device_index,
        queue,
        done,
        output_dir,
        chunk_length_seconds=0.5,
        rate=16000,
        frames_per_read=512,
        format=pyaudio.paInt16,
    ):
        # create recorder
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_read,
            input_device_index=device_index,
        )
        counter = 0

        while not done.value:
            # read chunk
            frames = []
            start_time = time.time()
            for i in range(0, int(rate / frames_per_read * chunk_length_seconds)):
                data = stream.read(frames_per_read)
                frames.append(data)
            stop_time = time.time()

            # save chunk to file
            next_file = output_dir / f"{counter:08}.wav"
            counter += 1
            wf = wave.open(str(next_file), "wb")
            wf.setnchannels(1)
            wf.setframerate(rate)
            wf.setsampwidth(p.get_sample_size(format))
            wf.writeframes(b"".join(frames))
            wf.close()

            # send audio file to queue
            queue.put(
                AudioFileInterface(
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=stop_time,
                    path=next_file,
                )
            )

        # stop recorder
        stream.stop_stream()
        stream.close()
        p.terminate()

        # clear queue
        while not queue.empty():
            queue.get()


# TODO: also may need to be modified to return a list of audio files if recording is faster than updates
@final
class RecordedAudio(BaseFeature[AudioFileInterface]):
    READ_FRAME_COUNT = 512
    SAVE_INTERVAL_SECONDS = 0.5

    def __init__(
        self,
        color_image: BaseFeature[ColorImageInterface],
        *,
        path: Path,
        speaker_id: str | None = None,
        video_frame_rate: int = 30,
    ):
        super().__init__(color_image)
        self.path = path
        self.video_frame_rate = video_frame_rate

        stub = f"recorded" + hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
        self.speaker_id = speaker_id if speaker_id is not None else stub
        self.output_dir = Path("chunks") / stub

    @property
    def audio_time(self):
        return self.num_frames_read / self.reader.getframerate()

    def save_if_needed(self):
        if self.audio_time - self.last_save_time >= self.SAVE_INTERVAL_SECONDS:
            next_file = self.output_dir / f"{self.counter:08}.wav"

            wf = wave.open(str(next_file), "wb")
            wf.setparams(self.reader.getparams())
            wf.writeframes(self.frames)
            wf.close()

            self.output.append(
                AudioFileInterface(
                    speaker_id=self.speaker_id,
                    start_time=time.time() - (self.audio_time - self.last_save_time),
                    end_time=time.time(),
                    path=next_file,
                )
            )

            self.last_save_time = self.audio_time
            self.frames = b""
            self.counter += 1

    def initialize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.reader = wave.open(str(self.path), "rb")
        self.num_frames_read = 0
        self.last_save_time = 0
        self.counter = 0
        self.frames = b""

        self.output = deque()

    def finalize(self):
        self.reader.close()

    def get_output(self, im: ColorImageInterface) -> AudioFileInterface | None:
        # while audio time < video time
        while self.audio_time < im.frame_count / self.video_frame_rate:
            self.frames += self.reader.readframes(self.READ_FRAME_COUNT)
            self.num_frames_read += self.READ_FRAME_COUNT

            self.save_if_needed()

        if len(self.output) == 0:
            return None

        return self.output.popleft()

    def is_done(self):
        return self.audio_time > self.reader.getnframes() / self.reader.getframerate()
