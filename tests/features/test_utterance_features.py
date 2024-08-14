import pytest


def test_recorded_input():
    assert False, "TODO"


@pytest.mark.parametrize(
    "input_map,output_lengths",
    [
        ("01101100110", [4, 4, 4]),
        ("11110100", [5, 3]),
        ("1111110", [5, 3]),  # check if max length works
        ("00001000001110", [3, 5]),
    ],
)
def test_vad_segmentation(input_map, output_lengths):
    """
    Arguments:
    `input_map` -- a string where 0 represents an audio clip with no voice
                   activity and 1 represents an audio clip with voice activity
    `output_lengths` -- a list of expected number of audio clips contained
                        inside the segmented utterances
    """
    # make audio input feature that sends chunks according to input_map
    # run demo with vad
    # have output feature which checks that:
    #   1. reported time of utterance is equal to the actual length
    #   2. time of utterance is equal to the expected output length
    #   3. there are no extra utterances
    assert False, "TODO"
