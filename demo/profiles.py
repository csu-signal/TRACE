"""
Profiles which can be used by the demo to load different devices
and evaluation configurations.
"""

import os
from typing import final

import demo.config as config
from demo.base_profile import BaseProfile, FrameInfo
from demo.config import K4A_DIR
from demo.featureModules import BaseDevice, MicDevice, PrerecordedDevice
from demo.featureModules.evaluation.eval_config import EvaluationConfig

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(K4A_DIR)
import azure_kinect


@final
class LiveProfile(BaseProfile):
    """
    A profile which uses the live camera and audio devices.

    Arguments:
    mic_info -- a list of tuples [(name, mic index), ...]
    """
    def __init__(self, mic_info: list[tuple[str, int]]):
        super().__init__(eval_config=None)
        self.mic_info = mic_info

    def create_camera_device(self):
        return azure_kinect.Camera(0)

    def create_audio_devices(self):
        return [MicDevice(i, j) for i,j in self.mic_info]

@final
class RecordedProfile(BaseProfile):
    """
    A profile which uses a prerecorded Azure Kinect mkv file and
    audio from a saved recording. Allows for passing evaluation info
    and early stopping of processing.

    Arguments:
    mkv_path -- the path to the mkv
    mic_info -- a list of tuples [(name, path to audio recording), ...]

    Keyword Arguments:
    eval_config -- the evaluation config to use (default None)
    mkv_frame_rate -- the frame rate of the mkv video (defualt 30)
    end_time -- the number of seconds at which to stop processing (default None)
    output_dir -- the directory to put the processing output into
    """
    def __init__(
        self,
        mkv_path: str,
        audio_info: list[tuple[str, str]],
        *,
        eval_config: EvaluationConfig | None = None,
        mkv_frame_rate=30,
        end_time: float | None = None,
        output_dir = None
    ):
        super().__init__(eval_config=eval_config, output_dir=output_dir)
        self.mkv = mkv_path
        self.audio_info = audio_info
        self.mkv_frame_rate = mkv_frame_rate

        self.audio_inputs = []

        self.end_frame = int(self.mkv_frame_rate * end_time) if end_time is not None else None

    def is_done(self, frame_count: int, fail_count: int):
        if self.end_frame is not None and frame_count > self.end_frame:
            return True

        return super().is_done(frame_count, fail_count)

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

def create_recorded_profile(path, *, output_dir=None, eval_config=None) -> RecordedProfile:
    """
    Create a profile from the output of the Azure Kinect Recorder.
    """
    return RecordedProfile(
        rf"{path}-master.mkv",
        [
            ("Group", rf"{path}-audio1.wav"),
            # ("Austin", rf"{path}-audio2.wav"),
            # ("Mariah", rf"{path}-audio3.wav"),
        ],
        eval_config=eval_config,
        output_dir=output_dir
    )


# TODO: remove later
@final
class BradyLaptopProfile(BaseProfile):
    """
    A profile that I (Brady) can use to test on my laptop. Uses the group
    1 mkv file and the laptop mic.
    """
    def __init__(self):
        super().__init__()

    def create_camera_device(self):
        return azure_kinect.Playback(r"C:\Users\brady\Desktop\Group_01-master.mkv")

    def create_audio_devices(self) -> list[BaseDevice]:
        return [MicDevice("Brady", 1)]

def create_wtd_eval_profiles(group, input_dir, output_dir, end_time=None) -> list[RecordedProfile]:
    """
    Create a profile to run WTD evaluation with ablation. It will run 4 profiles with no ground truth,
    asr ground truth, gesture ground truth, and object ground truth.

    Arguments:
    group -- which group to evaluation
    input_dir -- the location of input csv files, these can be created with the
                 "create_all_wtd_inputs.py" script
    output_dir -- output of each processing is stored in <output_dir>/group<group>/
    end_time -- the number of seconds in the recording after which the processing should stop
    """
    mkv = config.WTD_MKV_PATH.format(group)
    audio = config.WTD_AUDIO_PATH.format(group)

    eval_config_kwargs  = {
            "no_gt": {},
            "asr_gt": {"asr": True},
            "gesture_gt": {"gesture": True},
            "object_gt": {"objects": True},
    }

    models_kwargs = {
        "prop_model" : config.WTD_PROP_MODEL_PATH.format(group),
        "move_model" : config.WTD_MOVE_MODEL_PATH.format(group)
    }

    profiles = []
    for name, kwargs in eval_config_kwargs.items():
        eval_config = EvaluationConfig(f"{input_dir}\\group{group}", **models_kwargs, **kwargs, fallback_audio=audio)
        prof = RecordedProfile(
                mkv,
                [(f"Group {group}", audio)],
                eval_config=eval_config,
                output_dir=f"{output_dir}\\group{group}\\{name}",
                end_time=end_time)
        profiles.append(prof)

    return profiles
