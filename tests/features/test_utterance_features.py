import pytest

from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder


def test_recorded_input():
    assert False, "TODO"


# fixure cannot have scope="module" because it stores
# an internal state
@pytest.fixture
def vad_builder():
    vad = VADUtteranceBuilder()
    vad.initialize()
    yield vad
    vad.finalize()


@pytest.mark.parametrize(
    "inputs,expected",
    [
        # single speaker
        (["01101100110"], [[4, 4, 4]]),
        (["11110100"], [[5, 3]]),
        (["1111110"], [[5, 3]]),  # check if max length works
        ("00001000001110", [3, 5]),
        # multiple speakers
        (["0110", "110110", "000"], [[4], [3, 4], []]),
        (["010110", "0111100", "01010101"], [[3, 4], [6], [3, 3, 3]]),
        (
            # check if max length works for multiple speakers
            ["011111111110", "11111110", "0011110010"],
            [[5, 5, 4], [5, 4], [5, 3]],
        ),
    ],
)
def test_vad_segmentation_multiple_speakers(vad_builder, inputs, expected):
    """
    Test that the segmentation works correctly for both single and multiple
    speakers.

    `input` -- a list of string where 0 represents an audio clip with no voice
               activity and 1 represents an audio clip with voice activity.
               The different strings in the list correspond to different
               speakers who speak simultaneously.
    `output_lengths` -- a list of lists of expected lengths of audio clips
                        contained inside the segmented utterances. Different
                        lists correspond to different speakers.
    """
    # make audio input feature that sends chunks according to input_map
    # run demo with vad
    # have output feature which checks that:
    #   1. reported time of utterance is equal to the actual length
    #   2. time of utterance is equal to the expected output length
    #   3. there are no extra utterances
