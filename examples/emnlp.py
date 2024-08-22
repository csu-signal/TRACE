import cv2 as cv
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.features.dense_paraphrasing.dense_paraphrasing_feature import (
    DenseParaphrasing,
)
from mmdemo.features.dense_paraphrasing.referenced_objects_feature import (
    ReferencedObjects,
)
from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.move.move_feature import Move
from mmdemo.features.objects.object_feature import Object
from mmdemo.features.output_frames.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.features.selected_objects.selected_objects_feature import SelectedObjects
from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.features.utterance.audio_input_features import MicAudio
from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder
from mmdemo.interfaces import ColorImageInterface, EmptyInterface


class ShowOutput(BaseFeature[EmptyInterface]):
    def get_output(self, frame: ColorImageInterface):
        if not frame.is_new():
            return None

        cv.imshow("", frame.frame)
        cv.waitKey(1)


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
    audio = MicAudio(device_id=1, speaker_id="group")
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
    output_frame = EMNLPFrame(color, gaze, gesture, selected_objects, cgt)

    # run demo and show output
    Demo(targets=[ShowOutput(output_frame)]).run()
