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
    DpipObject,
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
        # Object Tracking -> DpipObject (Jack?)
            # TODO Update to include the new class values for the DPIP blocks
        # Propositions and Friction -> DpipProposition (Videep, Abhijnan)
            #TODO Update to return the new prop/friction output (interventions and structure status)
            #TODO Update TARSKI to include a script with Abhijnan's latest code to send back the DPIP values
        # CGT -> DpipCommonGroundTracking
            # TODO parse structure state from the friction output and render the sides
        # Output Frame -> DpipFrame

    # meeting questions
        # Videep Abhijnan utterance annotations column values? WTD 4, DPIP 3
            # G:\Weights_Task\Data\GAMR\Utterances
            # G:\DPIP\GAMR\Utterances 


# DPIP_MKV_PATH = (
#     "G:/DPIP/DPIP_Azure_Recordings/Group_Test_{0:02}-master.mkv"
# )

DPIP_MKV_PATH = (
    "D:/DPIP/DPIP_Azure_Recordings/SK_DPIP_Group_07-master.mkv"
)

DPIP_SECOND_MKV_PATH = (
    ""
)

# audio path for DPIP group
#DPIP_AUDIO_PATH = "G:/DPIP/DPIP_Azure_Recordings/Group_Test_{0:02}-audio1.wav"
DPIP_AUDIO_PATH ="D:/DPIP/DPIP_Azure_Recordings/SK_DPIP_Group_01-audio.mav"

# ground truth path for WTD group. These can be generated with
# scripts/dpip_annotations/create_all_dpip_inputs.py
DPIP_GROUND_TRUTH_DIR = "G:/DPIP/dpip_inputs/group{0:01}"
#DPIP_GROUND_TRUTH_DIR = "G:/DPIP/dpip_inputs/group3"

# DPIP_MOVE_MODEL_PATH = "G:/brady_wtd_eval_models/move_classifier_{0:02}.pt" #TODO: New(?) move model

# The number of seconds of the recording to process
# TODO update for DPIP (fine for now, but they might end early and not work for ablation testing)
DPIP_END_TIMES = {
    1: 5 * 60 + 30,
    2: 5 * 60 + 48,
    3: 8 * 60 + 3,
    4: 3 * 60 + 31,
    5: 4 * 60 + 34,
    6: 5 * 60 + 3,
    7: 4682,
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
    group = 7
    ground_truth_dir = Path(DPIP_GROUND_TRUTH_DIR.format(group))

    # load azure kinect features from file
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=Path(DPIP_MKV_PATH.format(group)),
        playback_end_seconds=DPIP_END_TIMES[group],
        playback_frame_rate=PLAYBACK_FRAME_RATE,
    )

    # load secondary azure kinect features from file
    # color2, depth2, body_tracking2, calibration2 = create_azure_kinect_features(
    #     DeviceType.PLAYBACK,
    #     mkv_path=Path(DPIP_SECOND_MKV_PATH.format(group)),
    #     playback_end_seconds=DPIP_END_TIMES[group],
    #     playback_frame_rate=PLAYBACK_FRAME_RATE,
    # )

    #gesture = Gesture(color, depth, body_tracking, calibration)
    # gesture2 = Gesture(color2, depth2, body_tracking2, calibration2)

    # which objects are selected by gesture
    # TODO update object info for new block types
    #objects = DpipObject(color, depth, calibration)
    #selected_objects = SelectedObjects(objects, gesture)

    # objects2 = DpipObject(color2, depth2, calibration2)
    # selected_objects2 = SelectedObjects(objects2, gesture2)

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
    #referenced_objects = AccumulatedSelectedObjects(selected_objects, transcriptions)
    # dense_paraphrased_transcriptions = DenseParaphrasedTranscription(
    #     transcriptions, referenced_objects
    # )

    #gesture_move = gesture
    # gesture_move = None
    #objects_move = selected_objects
    # objects_move = None

    # prop extraction from friction model
    dpip_prop_friction = DpipProposition(transcriptions, csvSupport="G:\\DPIP\\GAMR\\Utterances\\group7_transcript.csv")

    # TODO are we using Move?
    # moves = Move(dense_paraphrased_transcriptions, utterance_audio, gesture, selected_objects) #live move
    # moves = Move(
    #     dense_paraphrased_transcriptions,
    #     utterances,
    #     gesture=gesture_move,
    #     objects=objects_move,
    #     model_path=Path(WTD_MOVE_MODEL_PATH.format(group)),
    # )

    cgt = DpipCommonGroundTracking(dpip_prop_friction)
    
    # TODO are need to update the planner?
    # plan = Planner(cgt)

    # output_frame = DpipFrame(color, gesture, selected_objects, calibration)
    # output_frame2 = DpipFrame(color2, gesture2, selected_objects2, calibration2)

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(color),
            cgt, #new common ground gui output
            # SaveVideo(output_frame, frame_rate=10),
            # DisplayFrame(output_frame2),
            # SaveVideo(output_frame2, frame_rate=10, video_name=2),
            #Log(friction, csv=True),
            #Log(transcriptions, stdout=True),
        ]
    )
    #demo.show_dependency_graph()
    demo.run()
