import time
import wave
from pathlib import Path
from typing import final

import numpy as np
import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.features.utterance.audio_input_features import MicAudio, RecordedAudio
from mmdemo.interfaces import AudioFileInterface, ColorImageInterface


def get_length(wav_path):
    """
    Helper function to get length of wav file
    """
    wf = wave.open(str(wav_path), "rb")
    length = wf.getnframes() / wf.getframerate()
    wf.close()
    return length


@pytest.fixture(params=["testing.wav"])
def audio_file(request, test_data_dir):
    file: Path = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    assert (
        get_length(file) >= 5
    ), "This test requires all files to be at least 5 seconds long"
    assert (
        get_length(file) < 90
    ), "This test requires all files to less than 90 seconds long"
    return file


@final
class FakeColor(BaseFeature[ColorImageInterface]):
    def get_output(self):
        return None


@pytest.fixture(params=[200, 100])
def video_frame_rate(request):
    """
    Video frame rates to test
    """
    return request.param


@pytest.fixture
def recorded_audio(audio_file, video_frame_rate):
    rec = RecordedAudio(FakeColor(), path=audio_file, video_frame_rate=video_frame_rate)
    rec.initialize()
    yield rec
    rec.finalize()


# don't test values right when a new chunk should be produced (muliples of 50 here)
# because the behavior of the recorder is undefined then
@pytest.mark.parametrize(
    "frame_count",
    [16, 60, 160, 310, 410, 99999],
)
def test_recorder(
    recorded_audio: RecordedAudio, frame_count, video_frame_rate, audio_file
):
    if frame_count / video_frame_rate > get_length(audio_file):
        recorded_audio.get_output(
            ColorImageInterface(frame_count=frame_count, frame=np.zeros((5, 5, 3)))
        )
        assert recorded_audio.is_done(), "Feature should exit if the file is complete"
        return

    # get all outputs at a set frame rate
    all_outputs: list[AudioFileInterface] = []
    rec_output = recorded_audio.get_output(
        ColorImageInterface(frame_count=frame_count, frame=np.zeros((5, 5, 3)))
    )
    while rec_output is not None:
        all_outputs.append(rec_output)
        rec_output = recorded_audio.get_output(
            ColorImageInterface(frame_count=frame_count, frame=np.zeros((5, 5, 3)))
        )

    assert (
        len(all_outputs)
        == frame_count / video_frame_rate // recorded_audio.SAVE_INTERVAL_SECONDS
    ), "The number of chunks is not what was expected by this frame"

    for i in all_outputs:
        length = get_length(i.path)

        assert i.end_time - i.start_time == pytest.approx(
            length, abs=0.01
        ), "The output length according to the interface does not equal the length of the file"

        assert recorded_audio.SAVE_INTERVAL_SECONDS == pytest.approx(
            length, abs=0.1
        ), "The chunk length is not equal to the output interval"


@pytest.fixture
def mic_audio():
    mic = MicAudio(device_id=1)
    mic.initialize()
    yield mic
    mic.finalize()


@pytest.mark.xfail(reason="Will fail if no audio device with index 1 is availible")
def test_mic(mic_audio):
    start_time = time.time()
    while time.time() - start_time < 5:
        if mic_audio.get_output() is not None:
            return
        time.sleep(0.1)

    assert False, "Mic should have recorded something after 5 seconds"
