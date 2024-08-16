import mmdemo.features as fs
from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo


class PrintFeature(BaseFeature):
    def get_output(self, arg):
        if not arg.is_new():
            return None
        print(arg)
        return arg


if __name__ == "__main__":
    mic = fs.MicAudio(device_id=1)

    utterances = fs.VADUtteranceBuilder(mic)

    transcription = PrintFeature(fs.WhisperTranscription(utterances))

    props = PrintFeature(fs.Proposition(transcription))

    moves = PrintFeature(fs.Move(transcription, utterances))

    cgt = fs.CommonGroundTracking(moves, props)

    Demo(targets=[cgt]).run()
