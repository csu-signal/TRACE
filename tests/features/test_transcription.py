import string
import time
import wave

import pytest

from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.interfaces import AudioFileInterface, TranscriptionInterface


def get_length(wav_path):
    """
    Helper function to get length of wav file
    """
    wf = wave.open(str(wav_path), "rb")
    length = wf.getnframes() / wf.getframerate()
    wf.close()
    return length


def levenshtein(s1, s2):
    """
    Edit distance using DP
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


@pytest.fixture(scope="module")
def transcription_feature():
    asr = WhisperTranscription()
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
