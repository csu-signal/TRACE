import cv2 as cv
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features import (
    GazeBodyTracking,
    Gesture,
    Object,
    SelectedObjects,
    ReferencedObjects,
    VADUtteranceBuilder,
    MicAudio,
    WhisperTranscription,
    DenseParaphrasing,
    Proposition,
    Move,
    CommonGroundTracking,
    EMNLPFrame,
)
from mmdemo.interfaces import ColorImageInterface, EmptyInterface


class ShowOutput(BaseFeature[EmptyInterface]):
    def get_output(self, frame: ColorImageInterface):
        if not frame.is_new():
            return None

        cv.imshow("output", frame.frame)
        cv.waitKey(1)

    def is_done(self):
        return cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) < 1


class PrintOutput(BaseFeature):
    def get_output(self, *args):
        if not all(i.is_new() for i in args):
            return None

        for i in args:
            print(i)
        print()


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
            ShowOutput(output_frame),
            PrintOutput(dense_paraphrased_transcriptions, props, moves),
        ]
    )
    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
