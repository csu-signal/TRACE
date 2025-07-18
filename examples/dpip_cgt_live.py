# from mmdemo.features.friction.friction_feature import Friction
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features
from mmdemo.features.friction.friction_feature import Friction
from mmdemo.demo import Demo
from mmdemo.features import (

    CommonGroundTracking,
    DpipCommonGroundTracking,

    EMNLPFrame,
    DpipFrame,

    DpipObject,
    Object,

    DpipProposition,
    Proposition,

    AccumulatedSelectedObjects,
    DenseParaphrasedTranscription,
    DisplayFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    MicAudio,
    Move,
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

    # color2, depth2, body_tracking2, calibration2 = create_azure_kinect_features(
    #     DeviceType.CAMERA, camera_index=1
    # )

    # gaze and gesture
    # gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)
    # gesture2 = Gesture(color2, depth2, body_tracking2, calibration2)

    # which objects are selected by gesture
    objects = DpipObject(color, depth, calibration)
    # objects2 = DpipObject(color2, depth2, calibration2)

    # transcriptions from microphone 

    # Multiple microphones - laptop
    audio1 = MicAudio(device_id=9, speaker_id="D1")
    audio2 = MicAudio(device_id=4, speaker_id="D2")
    audio3 = MicAudio(device_id=15, speaker_id="D3")
    audio4 = MicAudio(device_id=3, speaker_id="Builder")
    utterance_audio = VADUtteranceBuilder(audio1, audio2, audio3, audio4, delete_input_files=False)

    #######################################################################################

    # single microphone - rosch microphone - Index: 39, Name: Microphone (USB audio CODEC)
    # audio1 = MicAudio(device_id=9, speaker_id="P1")
    # utterance_audio = VADUtteranceBuilder(audio1, delete_input_files=True)
    #######################################################################################

    transcriptions = WhisperTranscription(utterance_audio)

    # which objects are referenced (by gesture) during a transcription
    # and dense paraphrased transcription
    # referenced_objects = AccumulatedSelectedObjects(selected_objects, transcriptions)
    # dense_paraphrased_transcriptions = DenseParaphrasedTranscription(
    #     transcriptions, referenced_objects
    # )


    # prop extraction and move classifier
    props = DpipProposition(transcriptions, objects)
    # moves = Move(dense_paraphrased_transcriptions, utterance_audio, gesture, selected_objects)

    # common ground tracking
    cgt = DpipCommonGroundTracking(props)

    # plan = Planner(cgt)

    # friction
    # friction = Friction(dense_paraphrased_transcriptions, plan)
    # create output frame for video
    output_frame = DpipFrame(color, objects, calibration) # removed gaze, plan, cgt, friction
    # output_frame2 = DpipFrame(color2, objects2, calibration2)

    # run demo and show output
    demo = Demo(
        targets=[
            #DisplayFrame(color),
            #SaveVideo(output_frame, frame_rate=2.2),
            DisplayFrame(output_frame),
            cgt, #new common ground gui output
            #SaveVideo(output_frame2, frame_rate=2.2, video_name = 2),
            # Log(dense_paraphrased_transcriptions, props, moves, friction, csv=True),
            #Log(dense_paraphrased_transcriptions, props, csv=True),
            # Log(transcriptions, stdout=True),
        ]
    )
    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
