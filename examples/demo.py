import pickle

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

import mmdemo.features as fs
from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo


class PrintFeature(BaseFeature):
    def initialize(self):
        self.c = 0

    def get_output(self, *args):
        if not all(i.is_new() for i in args):
            return None

        with open(f"frame{self.c:05}.pkl", "wb") as f:
            pickle.dump(args, f)

        self.c += 1


if __name__ == "__main__":
    color, depth, bt, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path="C:\\Users\\brady\\Desktop\\Group_01-master.mkv",
        mkv_frame_rate=120,
        playback_frame_rate=1,
    )

    # mic = fs.MicAudio(device_id=1)
    #
    # utterances = fs.VADUtteranceBuilder(mic)
    #
    # transcription = PrintFeature(fs.WhisperTranscription(utterances))
    #
    # props = PrintFeature(fs.Proposition(transcription))
    #
    # moves = PrintFeature(fs.Move(transcription, utterances))
    #
    # cgt = fs.CommonGroundTracking(moves, props)

    Demo(targets=[PrintFeature(color, depth, bt, calibration)]).run()
