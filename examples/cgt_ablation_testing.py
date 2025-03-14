import os
from pathlib import Path
import pandas as pd

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    AccumulatedSelectedObjects,
    CommonGroundTracking,
    DenseParaphrasedTranscription,
    DisplayFrame,
    EMNLPFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    Move,
    Object,
    Proposition,
    RecordedAudio,
    SaveVideo,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
    Planner,
)
from mmdemo.features.wtd_ablation_testing import (
    GestureSelectedObjectsGroundTruth,
    ObjectGroundTruth,
    create_transcription_and_audio_ground_truth_features,
)

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# mkv path for WTD group
WTD_MKV_PATH = (
    "D:/Weights_Task/Data/Fib_weights_original_videos/Group_{0:02}-master.mkv"
)

# audio path for WTD group
WTD_AUDIO_PATH = "D:/Weights_Task/Data/Group_{0:02}-audio.wav"

# ground truth path for WTD group. These can be generated with
# scripts/wtd_annotations/create_all_wtd_inputs.py
WTD_GROUND_TRUTH_DIR = "wtd_inputs/group{0}"

# paths to models not trained on the current group
WTD_MOVE_MODEL_PATH = "D:/brady_wtd_eval_models/move_classifier_{0:02}.pt"
WTD_PROP_MODEL_PATH = "D:/brady_wtd_eval_models/steroid_model_g{0}"

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


def create_demo(
    group,
    *,
    ground_truth_objects=False,
    ground_truth_gestures=False,
    ground_truth_utterances=False,
    output_video_name=None,
):
    ground_truth_dir = Path(WTD_GROUND_TRUTH_DIR.format(group))

    # load azure kinect features from file
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=Path(WTD_MKV_PATH.format(group)),
        playback_end_seconds=WTD_END_TIMES[group],
        playback_frame_rate=PLAYBACK_FRAME_RATE,
    )

    if ground_truth_utterances:
        (
            transcriptions,
            utterances,
        ) = create_transcription_and_audio_ground_truth_features(
            color,
            csv_path=ground_truth_dir / "utterances.csv",
            chunk_dir_path=ground_truth_dir / "chunks",
        )
    else:
        # convert audio to a format that python can read
        # (sometimes the original one does not load correctly
        #  because of the encoding )
        converted_audio_file = Path(f"audio_converted_group{group}.wav")
        if not converted_audio_file.is_file():
            convert_audio(Path(WTD_AUDIO_PATH.format(group)), converted_audio_file)

        audio = RecordedAudio(color, path=converted_audio_file)
        utterances = VADUtteranceBuilder(audio, max_utterance_time=2.9)
        transcriptions = WhisperTranscription(utterances)

    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    if ground_truth_objects:
        objects = ObjectGroundTruth(
            depth, calibration, csv_path=ground_truth_dir / "objects.csv"
        )
    else:
        objects = Object(color, depth, calibration)

    if ground_truth_gestures:
        selected_objects = GestureSelectedObjectsGroundTruth(
            color, csv_path=ground_truth_dir / "gestures.csv"
        )
    else:
        selected_objects = SelectedObjects(objects, gesture)

    referenced_objects = AccumulatedSelectedObjects(selected_objects, transcriptions)

    dense_paraphrased_transcriptions = DenseParaphrasedTranscription(
        transcriptions, referenced_objects
    )

    props = Proposition(
        dense_paraphrased_transcriptions,
        model_path=Path(WTD_PROP_MODEL_PATH.format(group)),
    )

    gesture_move = gesture
    # gesture_move = None
    objects_move = selected_objects
    # objects_move = None

    moves = Move(
        dense_paraphrased_transcriptions,
        utterances,
        gesture=gesture_move,
        objects=objects_move,
        model_path=Path(WTD_MOVE_MODEL_PATH.format(group)),
    )

    cgt = CommonGroundTracking(moves, props)

    plan = Planner(cgt)

    output_frame = EMNLPFrame(color, gesture, selected_objects, cgt, calibration, plan)

    return Demo(
        targets=[
            DisplayFrame(output_frame),
            SaveVideo(
                output_frame,
                frame_rate=PLAYBACK_FRAME_RATE,
                video_name=output_video_name,
            ),
            Log(dense_paraphrased_transcriptions, props, moves, cgt, csv=True),
            Log(transcriptions, stdout=True),
        ]
    )


def convert_audio(input_path: Path, output_path: Path):
    """
    Ensure the input audio will be in the correct format using ffmpeg.
    Also normalizes the loudness of the audio.
    """
    os.system(
        f"ffmpeg -i {input_path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}"
    )


def add_audio_to_video(video, audio, output):
    """
    Add audio to a video using ffmpeg
    """
    os.system(
        f"ffmpeg -i {video} -i {audio} -map 0:v -map 1:a -c:v copy -shortest {output}"
    )


if __name__ == "__main__":

    group = 1
    output_frame_path = Path(f"output_frames_group{group}.mp4")
    final_video_path = Path(f"final_group{group}.mp4")

    # create demo with ground truth inputs
    demo = create_demo(
        group,
        ground_truth_objects=False,
        ground_truth_gestures=False,
        ground_truth_utterances=False,
        output_video_name=output_frame_path,
    )

    # demo.show_dependency_graph()

    demo.run()

    # add audio to output frames
    converted_audio_path = Path(f"audio_converted_group{group}.wav")
    backup_audio_path = Path(f"wtd_inputs/group{group}/chunks/full_recording.wav")
    audio_to_add = converted_audio_path if converted_audio_path.is_file() else backup_audio_path
    add_audio_to_video(
        output_frame_path, audio_to_add, final_video_path
    )

    demo.print_time_benchmarks()
