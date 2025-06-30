from mmdemo.base_feature import BaseFeature
from mmdemo.features.friction.friction_feature import Friction
from mmdemo.features.wtd_ablation_testing.transcription_feature import _TranscriptionAndAudioGroundTruth, AudioGroundTruth, TranscriptionGroundTruth
from mmdemo.interfaces import ColorImageInterface
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features
from pathlib import Path
import os

from mmdemo.demo import Demo
from mmdemo.features import (
    AccumulatedSelectedObjects,
    DpipCommonGroundTracking,
    DenseParaphrasedTranscription,
    DisplayFrame,
    DpipFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    MicAudio,
    RecordedAudio,
    Move,
    Object,
    DpipProposition,
    SaveVideo,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
    Planner,
    DpipCommonGroundTracking
)

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

#TODO 
    # Updated features needed (can be place holders for the time being)
        # Object Tracking
            # I think we can use the existing feature?
            # TODO to update/create a new GamrTargets/ObjectInfo to include the new class values for the DPIP blocks
        # Propositions -> DpipProposition
            #TODO Update to use the new propsition model
        # CGT -> DpipCommonGroundTracking
            # TODO output the bank values for the planner
            # TODO Dynamic Block Rendering/Updating (Hannah) (independent of the props output so we can parse and pass outputs from the model in when Videep is ready)
        # Output Frame -> DpipFrame

    # Get Post Processing working with DPIP
        # ground truth inputs (audio and others?) (Austin)
            # scripts/wtd_annotations/create_all_wtd_inputs.py
        # video end times?
        # once post process is working we can move into the output frame for the DPIP task

# TODO update to use the DPIP vidoes
# mkv path for WTD group
WTD_MKV_PATH = (
    "G:/Weights_Task/Data/Fib_weights_original_videos/Group_{0:02}-master.mkv"
)

# audio path for WTD group
WTD_AUDIO_PATH = "G:/Weights_Task/Data/Group_{0:02}-audio.wav"

# ground truth path for WTD group. These can be generated with
# scripts/wtd_annotations/create_all_wtd_inputs.py
WTD_GROUND_TRUTH_DIR = "G:/Weights_Task/Data/wtd_inputs/group{0}"

WTD_MOVE_MODEL_PATH = "G:/brady_wtd_eval_models/move_classifier_{0:02}.pt"

# The number of seconds of the recording to process
WTD_END_TIMES = {
    1: 5 * 60 + 30,
    2: 5 * 60 + 48,
    3: 8 * 60 + 3,
    4: 3 * 60 + 31,
    5: 4 * 60 + 34,
    6: 5 * 60 + 3,
    7: 8 * 60 + 30,
    8: 6 * 60 + 28,
    9: 3 * 60 + 46,
    10: 6 * 60 + 51,
    11: 2 * 60 + 19,
}

# Number of frames to evaluate per second. This must
# be a divisor of 30 (the true frame rate). Higher rates
# will take longer to process.
PLAYBACK_FRAME_RATE = 5

def convert_audio(input_path: Path, output_path: Path):
    """
    Ensure the input audio will be in the correct format using ffmpeg.
    Also normalizes the loudness of the audio.
    """
    os.system(
        f"ffmpeg -i {input_path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}"
    )

def create_transcription_and_audio_ground_truth_features(
    color: BaseFeature[ColorImageInterface], *, csv_path: Path, chunk_dir_path: Path):
    """
    Create features for transcription and audio ablation.

    Arguments:
    `color` -- feature which return color frames, used for frame count
    `csv_path` -- path to the WTD annotation utterances.csv file.
    `chunk_dir_path` -- path to the WTD annotation chunks directory

    Returns:
    transcription -- transcription feature which returns TranscriptionInterface
    audio -- audio feature which returns AudioFileInterface
    """
    ta = _TranscriptionAndAudioGroundTruth(
        color, csv_path=csv_path, chunk_dir_path=chunk_dir_path
    )
    transcription = TranscriptionGroundTruth(ta)
    audio = AudioGroundTruth(ta)

    return transcription, audio

if __name__ == "__main__":
    group = 1
    ground_truth_dir = Path(WTD_GROUND_TRUTH_DIR.format(group))

    # load azure kinect features from file
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=Path(WTD_MKV_PATH.format(group)),
        playback_end_seconds=WTD_END_TIMES[group],
        playback_frame_rate=PLAYBACK_FRAME_RATE,
    )


    # gaze and gesture
    # gaze = GazeBodyTracking(body_tracking, calibration) #TODO are we using gaze?
    gesture = Gesture(color, depth, body_tracking, calibration)

    # which objects are selected by gesture
    # TODO update object info for new block types
    objects = Object(color, depth, calibration)
    selected_objects = SelectedObjects(objects, gesture)

    #TODO get DPIP ground truth utterances
    # transcriptions from the ground truth file
    (
        transcriptions,
        utterances,
    ) = create_transcription_and_audio_ground_truth_features(
        color,
        csv_path=ground_truth_dir / "utterances.csv",
        chunk_dir_path=ground_truth_dir / "chunks",
    )

    # which objects are referenced (by gesture) during a transcription
    # and dense paraphrased transcription
    # TODO update selected objects info to return DPIP gamr values and not the old WTD gamr values
    referenced_objects = AccumulatedSelectedObjects(selected_objects, transcriptions)
    dense_paraphrased_transcriptions = DenseParaphrasedTranscription(
        transcriptions, referenced_objects
    )

    gesture_move = gesture
    # gesture_move = None
    objects_move = selected_objects
    # objects_move = None

    # prop extraction and move classifier
    props = DpipProposition(dense_paraphrased_transcriptions)

    # TODO are we using Move?
    # moves = Move(dense_paraphrased_transcriptions, utterance_audio, gesture, selected_objects) #live move
    # moves = Move(
    #     dense_paraphrased_transcriptions,
    #     utterances,
    #     gesture=gesture_move,
    #     objects=objects_move,
    #     model_path=Path(WTD_MOVE_MODEL_PATH.format(group)),
    # )

    cgt = DpipCommonGroundTracking(props)
    
    # TODO are need to update the planner?
    plan = Planner(cgt)

    # friction
    friction = Friction(dense_paraphrased_transcriptions, plan)

    output_frame = DpipFrame(color, gesture, selected_objects, calibration, friction, plan)

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            cgt, #new common ground gui output
            SaveVideo(output_frame, frame_rate=10),
            #Log(friction, csv=True),
            #Log(transcriptions, stdout=True),
        ]
    )
    #demo.show_dependency_graph()
    demo.run()
