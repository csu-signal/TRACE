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

        if self.c in [0, 197, 359, 1073]:
            with open(f"frame{self.c:05}.pkl", "wb") as f:
                pickle.dump(args, f)

        self.c += 1
    
    def is_done(self) -> bool:
        return self.c > 1100


if __name__ == "__main__":
    color, depth, bt, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=r"F:\brady_recording_tests\full_run_7_26\run03-master.mkv",
        mkv_frame_rate=30,
        playback_frame_rate=30,
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
