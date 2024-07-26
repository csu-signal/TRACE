"""
Profiles which can be used by the demo to load different devices
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import final

from config import K4A_DIR
from fake_camera import FakeCamera
from featureModules import BaseDevice, MicDevice, PrerecordedDevice
from base_profile import BaseProfile
from featureModules.evaluation.eval_config import EvaluationConfig

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(K4A_DIR)
import azure_kinect

@final
class LiveProfile(BaseProfile):
    def __init__(self, mic_info: list[tuple[str, int]]):
        super().__init__(eval_config=None)
        self.mic_info = mic_info

    def create_camera_device(self):
        return azure_kinect.Camera(0)

    def create_audio_devices(self):
        return [MicDevice(i, j) for i,j in self.mic_info]

@final
class RecordedProfile(BaseProfile):
    def __init__(
        self,
        mkv_path: str,
        audio_info: list[tuple[str, str]],
        eval_config: EvaluationConfig | None = None,
        mkv_frame_rate=30,
    ):
        super().__init__(eval_config)
        self.mkv = mkv_path
        self.audio_info = audio_info
        self.mkv_frame_rate = mkv_frame_rate

        self.audio_inputs = []

    def create_camera_device(self):
        return azure_kinect.Playback(self.mkv)

    def convert_audio(self, path, index):
        output_path = f"{self.video_dir}\\audio{index}.wav"
        os.system(f"ffmpeg -i {path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}")
        self.audio_inputs.append(output_path)
        return output_path

    def create_audio_devices(self) -> list[BaseDevice]:
        return [
                PrerecordedDevice(name, self.convert_audio(file, index), self.mkv_frame_rate)
                for index, (name,file) in enumerate(self.audio_info)
                ]

    def finalize(self):
        # turn frames into video
        super().finalize()

        num_audio = len(self.audio_inputs)
        audio_inputs = " ".join([f"-i {file}" for file in self.audio_inputs])

        if num_audio > 0:
            # combine all audio recordings
            os.system(f"ffmpeg {audio_inputs} -filter_complex amix=inputs={num_audio}:duration=shortest {self.video_dir}\\audio-combined.wav")

            # add audio to video
            os.system(f"ffmpeg -i {self.video_dir}\\processed_frames.mp4 -i {self.video_dir}\\audio-combined.wav -map 0:v -map 1:a -c:v copy -shortest {self.video_dir}\\final.mp4")
        elif self.eval is not None and self.eval.fallback_audio is not None:
            os.system(f"ffmpeg -i {self.video_dir}\\processed_frames.mp4 -i {self.eval.fallback_audio} -map 0:v -map 1:a -c:v copy -shortest {self.video_dir}\\final.mp4")

        print(f"saved video as {self.video_dir}\\final.mp4")

def create_recorded_profile(path, eval_config=None):
    return RecordedProfile(
        rf"{path}-master.mkv",
        [
            ("Group", rf"{path}-audio1.wav"),
            # ("Austin", rf"{path}-audio2.wav"),
            # ("Mariah", rf"{path}-audio3.wav"),
        ],
        eval_config
    )


# TODO: remove later
@final
class BradyLaptopProfile(BaseProfile):
    def __init__(self):
        super().__init__(eval_config=None)

    def create_camera_device(self):
        return azure_kinect.Playback(r"C:\Users\brady\Desktop\Group_01-master.mkv")

    def create_audio_devices(self) -> list[BaseDevice]:
        return [MicDevice("Brady", 1)]

@final
class TestDenseParaphrasingProfile(BaseProfile):
    def __init__(self) -> None:
        super().__init__(EvaluationConfig(directory="test_inputs\\dense_paraphrasing", asr=True, gesture=True))

    def create_camera_device(self):
        return FakeCamera()

    def create_audio_devices(self):
        return []
