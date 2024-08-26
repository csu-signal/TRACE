from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (CommonGroundTracking, DenseParaphrasing,
                             DisplayFrame, EMNLPFrame, GazeBodyTracking,
                             Gesture, Log, MicAudio, Move, Object, Proposition,
                             ReferencedObjects, SelectedObjects,
                             VADUtteranceBuilder, WhisperTranscription)

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
    referenced_objects = ReferencedObjects(selected_objects, transcriptions)
    dense_paraphrased_transcriptions = DenseParaphrasing(
        transcriptions, referenced_objects
    )

    # prop extraction and move classifier
    props = Proposition(dense_paraphrased_transcriptions)
    moves = Move(dense_paraphrased_transcriptions, utterance_audio)

    # common ground tracking
    cgt = CommonGroundTracking(moves, props)

    # create output frame for video
    output_frame = EMNLPFrame(color, gaze, gesture, selected_objects, cgt, calibration)

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            Log(dense_paraphrased_transcriptions, props, moves, csv=True),
            Log(transcriptions, dense_paraphrased_transcriptions, stdout=True)
        ]
    )
    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()