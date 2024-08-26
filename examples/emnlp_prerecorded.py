from pathlib import Path
import os

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
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
    AccumulatedSelectedObjects,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
)


def convert_audio(input_path, output_path):
    os.system(
        f"ffmpeg -i {input_path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}"
    )


if __name__ == "__main__":
    # azure kinect features from camera
    mkv_path = Path(
        rf"F:\Weights_Task\Data\Fib_weights_original_videos\Group_01-master.mkv"
    )

    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK, mkv_path=mkv_path, playback_end_seconds=5 * 60 + 30
    )

    audio_path = Path("audio_wtd_group01.wav")
    convert_audio(Path(rf"F:\Weights_Task\Data\Group_01-audio.wav"), audio_path)

    audio = RecordedAudio(color, path=audio_path)
    vad_utt_audio = VADUtteranceBuilder(audio, delete_input_files=True)
    transcriptions = WhisperTranscription(vad_utt_audio)

    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    obj = Object(color, depth, calibration)
    sel_obj = SelectedObjects(obj, gesture)

    ref_obj = AccumulatedSelectedObjects(sel_obj, transcriptions)

    dense = DenseParaphrasedTranscription(transcriptions, ref_obj)

    prop = Proposition(dense)
    move = Move(dense, vad_utt_audio)

    cgt = CommonGroundTracking(move, prop)

    output = EMNLPFrame(color, gaze, gesture, sel_obj, cgt, calibration)

    demo = Demo(
        targets=[
            DisplayFrame(output),
            Log(dense, prop, move, csv=True),
            Log(transcriptions, stdout=True)
        ]
    )

    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
