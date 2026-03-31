import os
import sys
import warnings

# Suppress noisy Google API Core Python version deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version .* google\.api_core",
    category=FutureWarning,
)

from mmdemo.features.friction.sensor_sheet_friction_feature import SensorSheetFrictionFeature
from mmdemo.features.outputs.sensor_frame_feature import SensorFrame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    AccumulatedSelectedObjects,
    CommonGroundTracking,
    DenseParaphrasedTranscription,
    DisplayFrame,
    DpipActionFeature,
    DpipBlockDetections,
    DpipCommonGroundTracking,
    DpipFrame,
    DpipObject,
    DpipProposition,
    EMNLPFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    MicAudio,
    Move,
    Object,
    Planner,
    Proposition,
    SaveVideo,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
)
from mmdemo.features.friction.friction_feature import Friction

# from mmdemo.features.friction.friction_feature import Friction
from mmdemo.features.speech_output.dpipSpeechoutput_feature import DpipSpeechOutput

if __name__ == "__main__":
    print(f"is cuda available? {torch.cuda.is_available()}")
    # azure kinect features from camera
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA, camera_index=0
    )

    # transcriptions from microphone

    # Multiple microphones - laptop
    audio1 = MicAudio(device_id=12, speaker_id="D1")
    audio2 = MicAudio(device_id=14, speaker_id="D2")
    audio3 = MicAudio(device_id=15, speaker_id="D3")
    audio4 = MicAudio(device_id=3, speaker_id="Builder")
    utterance_audio = VADUtteranceBuilder(
        audio1, audio2, audio3, audio4, delete_input_files=False
    )

    #######################################################################################

    # single microphone - rosch microphone - Index: 39, Name: Microphone (USB audio CODEC)
    # audio1 = MicAudio(device_id =9, speaker_id="P1")
    # utterance_audio = VADUtteranceBuilder(audio1, delete_input_files=True)
    #######################################################################################

    transcriptions = WhisperTranscription(utterance_audio)

    # TODO create feature to get data from sheet
    # TODO create feature to call into sensor activity friction LLM
    friction = SensorSheetFrictionFeature(transcriptions)

    # create output frame for video
    output_frame = SensorFrame(color, friction)

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            SaveVideo(output_frame, frame_rate=2.2, video_name=2),
            Log(transcriptions, csv=True)
        ]
    )
    # demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
