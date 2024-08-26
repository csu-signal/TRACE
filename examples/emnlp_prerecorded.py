from pathlib import Path

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    CommonGroundTracking,
    DenseParaphrasing,
    DisplayFrame,
    EMNLPFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    Move,
    Object,
    Proposition,
    RecordedAudio,
    ReferencedObjects,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
)

if __name__ == "__main__":
    # azure kinect features from camera
    mkv_path = Path(rf"..\videos\videoName.mkv")

    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK, mkv_path=mkv_path
    )

    audio = RecordedAudio(
        color,
    )
    vad_utt_audio = VADUtteranceBuilder(audio, delete_input_files=True)
    transcriptions = WhisperTranscription(vad_utt_audio)

    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    obj = Object(color, depth, calibration)
    sel_obj = SelectedObjects(obj, gesture)

    ref_obj = ReferencedObjects(sel_obj, transcriptions)

    dense = DenseParaphrasing(transcriptions, sel_obj)

    prop = Proposition(transcriptions)
    move = Move(transcriptions, audio)

    cgt = CommonGroundTracking(move, prop)

    output = EMNLPFrame(color, gaze, gesture, sel_obj, cgt, calibration)

    demo = Demo(
        targets=[
            DisplayFrame(color),
            Log(dense, prop, move, csv=True),
            Log(transcriptions, dense, stdout=True),
        ]
    )

    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
