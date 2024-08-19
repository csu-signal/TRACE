import pytest

from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.interfaces import CommonGroundInterface, MoveInterface, PropositionInterface


@pytest.fixture(scope="module")
def cgt_feature():
    """
    Uses the same feature for all tests in this file,
    so the order that tests are run will matter because an
    internal state is stored
    """
    cgt = CommonGroundTracking()
    cgt.initialize()
    yield cgt
    cgt.finalize()


# TODO: See if Nikhil can write tests for these:
#   - multiple moves at the same time
#   - accepts / doubts with no previous statement
#   - !=, >, <, >=, <=
@pytest.mark.parametrize(
    "prop, move, ebank, fbank",
    [
        ("red = 10", {"STATEMENT"}, {"red=10"}, set()),
        ("red = 10", {"ACCEPT"}, set(), {"red=10"}),
        ("blue = 10", {"STATEMENT"}, {"blue=10"}, {"red=10"}),
        ("blue = 20", {"STATEMENT"}, {"blue=10", "blue=20"}, {"red=10"}),
        ("blue = 20", {"ACCEPT"}, set(), {"red=10", "blue=20"}),
        ("blue = 20", {"DOUBT"}, {"blue=20"}, {"red=10"}),
        ("blue = 10", {"STATEMENT"}, {"blue=10", "blue=20"}, {"red=10"}),
        ("blue = 10", {"ACCEPT"}, set(), {"red=10", "blue=10"}),
    ],
)
def test_cgt_feature(cgt_feature: CommonGroundTracking, prop, move, ebank, fbank):
    """
    Test that common ground tracking works correctly.
    """
    output = cgt_feature.get_output(
        MoveInterface(speaker_id="test", move=move),
        PropositionInterface(speaker_id="test", prop=prop),
    )
    assert isinstance(output, CommonGroundInterface)

    assert output.ebank == ebank
    assert output.fbank == fbank
