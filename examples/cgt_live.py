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
    # gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    # which objects are selected by gesture
    objects = Object(color, depth, calibration)
    selected_objects = SelectedObjects(objects, gesture)  # pyright: ignore

    # transcriptions from microphone 

    # laptop microphones
    # audio1 = MicAudio(device_id=7, speaker_id="P1")
    # audio2 = MicAudio(device_id=9, speaker_id="P2")
    # audio3 = MicAudio(device_id=11, speaker_id="P3")
    # utterance_audio = VADUtteranceBuilder(audio1, audio2, audio3, delete_input_files=False)

    audio = MicAudio(device_id=7, speaker_id="P1")
    utterance_audio = VADUtteranceBuilder(audio, delete_input_files=False)

    #######################################################################################

    # rosch microphone - Index: 39, Name: Microphone (USB audio CODEC)
    # audio1 = MicAudio(device_id=9, speaker_id="P1")
    # utterance_audio = VADUtteranceBuilder(audio1, delete_input_files=True)
    #######################################################################################

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

    # common ground tracking
    cgt = CommonGroundTracking(moves, props)

    plan = Planner(cgt)

    # friction
    friction = Friction(dense_paraphrased_transcriptions, plan)
    # create output frame for video
    output_frame = EMNLPFrame(color, gesture, selected_objects, cgt, calibration, friction, plan) #removed gaze

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            SaveVideo(output_frame, frame_rate=2.2),
            Log(dense_paraphrased_transcriptions, props, moves, friction, csv=True),
            # Log(transcriptions, stdout=True),
        ]
    )
    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
