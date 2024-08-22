import string
import time

import pytest

from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.interfaces import AudioFileInterface, TranscriptionInterface
from tests.utils.audio import get_length
from tests.utils.fake_feature import FakeFeature
from tests.utils.text import levenshtein


@pytest.fixture(scope="module")
def transcription_feature():
    asr = WhisperTranscription(FakeFeature())
    asr.initialize()
    yield asr
    asr.finalize()


@pytest.mark.parametrize(
    "audio_file,expected_transcription",
    [("testing.wav", "this is the first test test two finally the third test")],
)
def test_whisper_transcription(
    transcription_feature: WhisperTranscription,
    audio_file,
    expected_transcription,
    test_data_dir,
):
    """
    Test that transcription works correctly. Audio files
    are realtive to `test_data_dir`. Expected transcriptions
    should be lowercase and with no punctuation.
    """
    full_path = test_data_dir / audio_file
    end_time = time.time()
    start_time = end_time - get_length(full_path)

    output = transcription_feature.get_output(
        AudioFileInterface(
            speaker_id="test", start_time=start_time, end_time=end_time, path=full_path
        )
    )
    assert isinstance(output, TranscriptionInterface)

    # clean text so it can be compared to expected_transcription
    filtered_text = output.text
    # remove punctuation: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    filtered_text = filtered_text.translate(str.maketrans("", "", string.punctuation))
    filtered_text = filtered_text.lower().strip()

    assert output.start_time == start_time
    assert output.end_time == end_time
    assert output.speaker_id == "test"

    # edit distance between strings should be within 10
    assert (
        levenshtein(filtered_text, expected_transcription) < 10
    ), "Output string is too different from the expected"
