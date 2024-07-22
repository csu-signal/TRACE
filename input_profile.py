"""
Profiles which can be used by the demo to load different devices
"""

from abc import ABC, abstractmethod
import os
from typing import final
import config
from datetime import datetime

from featureModules import BaseDevice, MicDevice, PrerecordedDevice

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(config.K4A_DIR)
import azure_kinect


class BaseProfile(ABC):
    def finalize(self):
        pass

    @abstractmethod
    def get_output_dir(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_camera_device(self) -> azure_kinect.Device:
        raise NotImplementedError

    @abstractmethod
    def get_audio_devices(self):
        raise NotImplementedError

@final
class LiveProfile(BaseProfile):
    def __init__(self, mic_info: list[tuple[str, int]]):
        self.mic_info = mic_info
        self.output_dir = f"stats_{str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))}"

    def get_camera_device(self):
        return azure_kinect.Camera(0)

    def get_audio_devices(self):
        return [MicDevice(i, j) for i,j in self.mic_info]

    def get_output_dir(self) -> str:
        return self.output_dir

@final
class RecordedProfile(BaseProfile):
    def __init__(self, mkv_path: str, audio_info: list[tuple[str, str]], mkv_frame_rate=30):
        self.mkv = mkv_path
        self.audio_info = audio_info
        self.mkv_fr = mkv_frame_rate

        self.audio_inputs = []

        self.output_dir = f"stats_{str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))}"
        self.video_dir = f"{self.output_dir}\\video_files"

    def get_camera_device(self):
        return azure_kinect.Playback(self.mkv)

    def convert_audio(self, path, index):
        os.makedirs(self.video_dir, exist_ok=True)
        output_path = f"{self.video_dir}\\audio{index}.wav"
        os.system(f"ffmpeg -i {path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}")
        self.audio_inputs.append(output_path)
        return output_path

    def get_audio_devices(self) -> list[BaseDevice]:
        return [
                PrerecordedDevice(name, self.convert_audio(file, index), video_frame_rate=self.mkv_fr)
                for index, (name,file) in enumerate(self.audio_info)
                ]

    def get_output_dir(self) -> str:
        return self.output_dir

    def finalize(self):
        # turn frames into video
        os.system(f"ffmpeg -framerate 30 -i {self.output_dir}\\processed_frames\\frame%8d.png -c:v libx264 -pix_fmt yuv420p {self.video_dir}\\processed_frames.mp4")
        os.system(f"ffmpeg -framerate 30 -i {self.output_dir}\\raw_frames\\frame%8d.png -c:v libx264 -pix_fmt yuv420p {self.video_dir}\\raw_frames.mp4")

        num_audio = len(self.audio_inputs)
        audio_inputs = " ".join([f"-i {file}" for file in self.audio_inputs])

        # combine all audio recordings
        os.system(f"ffmpeg {audio_inputs} -filter_complex amix=inputs={num_audio}:duration=shortest {self.video_dir}\\audio-combined.wav")

        # add audio to video
        os.system(f"ffmpeg -i {self.video_dir}\\processed_frames.mp4 -i {self.video_dir}\\audio-combined.wav -map 0:v -map 1:a -c:v copy -shortest {self.video_dir}\\final.mp4")

        print(f"saved video as {self.video_dir}\\final.mp4")

def create_recorded_profile(path):
    return RecordedProfile(
        rf"{path}-master.mkv",
        [
            ("Group", rf"{path}-audio1.wav"),
            # ("Austin", rf"{path}-audio2.wav"),
            # ("Mariah", rf"{path}-audio3.wav"),
        ],
    )
