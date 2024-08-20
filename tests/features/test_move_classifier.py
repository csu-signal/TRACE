import pytest

from mmdemo.features.move.move_feature import Move
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.interfaces import (
    AudioFileInterface,
    MoveInterface,
    PropositionInterface,
    TranscriptionInterface,
)


@pytest.fixture(scope="module")
def move_feature():
    """
    Uses the same move feature for all tests in this file,
    so the order that tests are run will matter because an
    internal state is stored
    """
    move = Move()
    move.initialize()
    yield move
    move.finalize()


# TODO: record actual inputs to have audio files
@pytest.mark.model_dependent
@pytest.mark.parametrize(
    "text,audio_file,expected",
    [
        ("red is 10", "testing.wav", ["STATEMENT"]),
        ("Yeah that is correct", "testing.wav", ["ACCEPT"]),
        ("Wait that seems wrong", "testing.wav", ["DOUBT"]),
        ("blue is 10", "testing.wav", ["STATEMENT"]),
        ("green is 10", "testing.wav", ["STATEMENT"]),
        ("We can see that purple is 10", "testing.wav", ["STATEMENT"]),
    ],
)
def test_move_classification(
    move_feature: Move, test_data_dir, text, audio_file, expected
):
    """
    Test that prop extraction works correctly.
    """
    info = {"speaker_id": "test", "start_time": 0, "end_time": 1}

    output = move_feature.get_output(
        TranscriptionInterface(**info, text=text),
        AudioFileInterface(**info, path=test_data_dir / audio_file),
    )
    assert isinstance(output, MoveInterface)

    assert list(sorted(output.move)) == list(sorted(expected))
