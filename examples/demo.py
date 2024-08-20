import pickle

import mmdemo.features as fs
from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.features.move.move_feature import Move
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.features.utterance.audio_input_features import MicAudio
from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder

# from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features


class PrintFeature(BaseFeature):
    def get_output(self, *args):
        print(args)


if __name__ == "__main__":
    mic = MicAudio(device_id=1)

    utterances = VADUtteranceBuilder(mic)

    transcription = WhisperTranscription(utterances)

    props = Proposition(transcription)

    moves = Move(transcription, utterances)

    cgt = CommonGroundTracking(moves, props)

    Demo(targets=[PrintFeature(transcription, props)]).run()
