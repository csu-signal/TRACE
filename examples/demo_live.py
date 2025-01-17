from mmdemo.features.friction.friction_feature import Friction
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
    MicAudio,
    Move,
    Object,
    Proposition,
    SaveVideo,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
    Planner,
)

if __name__ == "__main__":
    # azure kinect features from camera
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA, camera_index=0
    )

    # gaze and gesture
    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    # which objects are selected by gesture
    objects = Object(color, depth, calibration)
    selected_objects = SelectedObjects(objects, gesture)  # pyright: ignore

    # transcriptions from microphone
    audio = MicAudio(device_id=6, speaker_id="group")
    utterance_audio = VADUtteranceBuilder(audio, delete_input_files=True)
    transcriptions = WhisperTranscription(utterance_audio)

    # which objects are referenced (by gesture) during a transcription
    # and dense paraphrased transcription
    referenced_objects = AccumulatedSelectedObjects(selected_objects, transcriptions)
    dense_paraphrased_transcriptions = DenseParaphrasedTranscription(
        transcriptions, referenced_objects
    )

    # prop extraction and move classifier
    props = Proposition(dense_paraphrased_transcriptions)
    moves = Move(dense_paraphrased_transcriptions, utterance_audio, gesture, selected_objects)

    # TODO create frictive freature
    # TODO pass in relevant inputs
    friction = Friction()

    # plan = Planner(cgt)

    # TODO create output frame for this demo
    #output_frame = EMNLPFrame(color, gaze, gesture, selected_objects, cgt, calibration)

    # run demo and show output
    demo = Demo(
        targets=[
            #DisplayFrame(output_frame),
            #SaveVideo(output_frame, frame_rate=10),
            #Log(dense_paraphrased_transcriptions, props, moves, csv=True),
            #Log(transcriptions, stdout=True),
        ]
    )
    demo.show_dependency_graph()
    demo.run()
