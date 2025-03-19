import hashlib
import multiprocessing as mp
import os
import shutil
import time
import wave
from collections import deque
from ctypes import c_bool
from pathlib import Path
from typing import final
import sys

import pyaudio

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    AudioFileInterface,
    AudioFileListInterface,
    ColorImageInterface,
)
from mmdemo.utils.files import create_tmp_dir_with_featureName


@final
class MicAudio(BaseFeature[AudioFileListInterface]):
    """
    Record audio chunks from a mic. The output of this feature
    will likely need to be passed through an utterance segmentation
    feature before being used.

    There is no input interface.

    The output interface is `AudioFileListInterface`. The list version
    of the interface is needed because if the demo is processing at a slower
    rate than chunk production, the audio will lag behind in the demo.

    Keyword arguments:
    `device_id` -- the index of the audio device to open
    `speaker_id` -- a unique identifier for the speaker of this audio
    """

    def __init__(self, *, device_id: int, delete_output_audio=True, speaker_id: str | None = None) -> None:
        super().__init__()
        self.device_id = device_id
        self.delete_output_audio = delete_output_audio
        self.speaker_id = speaker_id if speaker_id is not None else f"mic{device_id:03}"

    def initialize(self):
        self.output_dir = create_tmp_dir_with_featureName(f"micAudio")

        self.queue = mp.Queue()
        self.done = mp.Value(c_bool, False)

        # start recorder process
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

        if self.delete_output_audio:
            shutil.rmtree(self.output_dir)

    def get_output(self) -> AudioFileListInterface | None:
        if self.queue.empty():
            return None

        # take files from the recorder queue and output them
        files = []
        while not self.queue.empty():
            files.append(self.queue.get())
        return AudioFileListInterface(audio_files=files)

    @staticmethod
    def record_chunks(
        speaker_id,
        device_index,
        queue,
        done,
        output_dir,
        # TODO: the performace seems very sensitive to chunk lengths
        # but the behavior is different based on how long people wait
        # between utterances. Some exploration may be needed to find
        # the optimal value, or a smarter utterance segmentation could
        # be implemented.
        chunk_length_seconds=1,
        rate=16000,
    ):
        """
        This method is run in a separate process to record
        chunks. This way there is no missed audio.

        Arguments:
        `speaker_id` -- an identifier for the speaker of the audio
        `device_id` -- the index of the audio device to open
        `queue` -- the output queue of the process
        `done` -- a boolean shared variable to mark when the process should exit
        `output_dir` -- the output directory of audio chunks

        Keyword arguments:
        `chunk_length_seconds` -- the length of each audio chunk (default 1)
        `rate` -- the audio rate in Hz (default 16000)
        """

        format = pyaudio.paInt16
        frames_per_read = 512

        try:
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
        except Exception as e:
            print(f"AUDIO INPUT FEATURE: An error occurred initalizing audio device {device_index}: {e} ")
            return

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


@final
class RecordedAudio(BaseFeature[AudioFileListInterface]):
    """
    Produce audio chunks from a wav file. The output of this feature
    will likely need to be passed through an utterance segmentation
    feature before being used.

    The input interface is `ColorImageInterface` (so the frame count can
    be accessed)

    The output interface is `AudioFileListInterface`. The list version
    of the interface is needed because if the demo is processing at a slower
    rate than chunk production, the audio will lag behind in the demo.

    Keyword arguments:
    `path` -- the path to the wav file
    `speaker_id` -- a unique identifier for the speaker of this audio
    `video_frame_rate` -- the frame rate of the video which is being used
        to get frame counts. This is needed to determine how much time has
        passed at the current frame in the video.
    """

    # the number of frames read at a time
    READ_FRAME_COUNT = 512

    # the length of output audio files
    # TODO: (copied from above) the performace seems very sensitive to chunk
    # lengths but the behavior is different based on how long people wait
    # between utterances. Some exploration may be needed to find the optimal
    # value, or a smarter utterance segmentation could be implemented.
    SAVE_INTERVAL_SECONDS = 1

    def __init__(
        self,
        color_image: BaseFeature[ColorImageInterface],
        *,
        path: Path,
        delete_output_audio=True,
        speaker_id: str | None = None,
        video_frame_rate: int = 30,
    ):
        super().__init__(color_image)
        self.path = path
        self.video_frame_rate = video_frame_rate
        self.delete_output_audio = delete_output_audio

        if speaker_id is not None:
            self.speaker_id = speaker_id
        else:
            self.speaker_id = (
                f"recorded" + hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
            )

    @property
    def audio_time(self):
        return self.num_frames_read / self.reader.getframerate()

    def initialize(self):
        # create output directory, open input file, and initialize
        # params
        self.output_dir = create_tmp_dir_with_featureName("recordedAudio")
        self.reader = wave.open(str(self.path), "rb")
        self.num_frames_read = 0
        self.last_save_time = 0
        self.counter = 0
        self.frames = b""

        self.output = deque()

    def finalize(self):
        self.reader.close()
        shutil.rmtree(self.output_dir)

    def save_if_needed(self):
        """
        Save a new audio chunk if enough frames have been accumulated
        """
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

    def get_output(self, im: ColorImageInterface) -> AudioFileListInterface | None:
        # while audio time < video time
        while self.audio_time < im.frame_count / self.video_frame_rate:
            # accumulate frames and save the current audio if
            # needed
            self.frames += self.reader.readframes(self.READ_FRAME_COUNT)
            self.num_frames_read += self.READ_FRAME_COUNT
            self.save_if_needed()

        if len(self.output) == 0:
            return None

        # return all saved audio files
        files = []
        while len(self.output) > 0:
            files.append(self.output.popleft())

        return AudioFileListInterface(audio_files=files)

    def is_done(self):
        # if the audio file is complete, the demo can exit
        return self.audio_time > self.reader.getnframes() / self.reader.getframerate()
