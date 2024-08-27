from mmdemo.demo import Demo
from mmdemo.features import (
    CommonGroundTracking,
    Log,
    MicAudio,
    Move,
    Proposition,
    VADUtteranceBuilder,
    WhisperTranscription,
)

if __name__ == "__main__":
    mic = MicAudio(device_id=6)
    utterances = VADUtteranceBuilder(mic)
    transcription = WhisperTranscription(utterances)
    props = Proposition(transcription)
    moves = Move(transcription, utterances)
    cgt = CommonGroundTracking(moves, props)

    demo = Demo(targets=[Log(mic, utterances, transcription, stdout=True)])
    demo.show_dependency_graph()
    demo.run()
    demo.print_time_benchmarks()
