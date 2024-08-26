import pytest

from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.interfaces import CommonGroundInterface, MoveInterface, PropositionInterface
from tests.utils.fake_feature import FakeFeature


@pytest.fixture(scope="module")
def cgt_feature():
    """
    Uses the same feature for all tests in this file,
    so the order that tests are run will matter because an
    internal state is stored
    """
    cgt = CommonGroundTracking(FakeFeature(), FakeFeature())
    cgt.initialize()
    yield cgt
    cgt.finalize()


# TODO: I am not sure what the expected behavior
# is for the following, but if we find out then we should
# add tests
#   - multiple moves at the same time
#   - accepts / doubts with no previous statement
#   - !=, >, <, >=, <=
@pytest.mark.parametrize(
    "prop, move, expected_ebank, expected_fbank",
    [
        ("red=10", {"STATEMENT"}, {"red=10"}, set()),
        ("red=10", {"ACCEPT"}, set(), {"red=10"}),
        ("blue=10", {"STATEMENT"}, {"blue=10"}, {"red=10"}),
        ("blue=20", {"STATEMENT"}, {"blue=10", "blue=20"}, {"red=10"}),
        ("blue=20", {"ACCEPT"}, set(), {"red=10", "blue=20"}),
        ("blue=20", {"DOUBT"}, {"blue=10", "blue=20"}, {"red=10"}),
        (
            "green>20",
            {"STATEMENT"},
            {"blue=10", "blue=20", "green!=10", "green!=20"},
            {"red=10"},
        ),
        (
            "green!=10",
            {"ACCEPT"},
            {"blue=10", "blue=20", "green!=20"},
            {"red=10", "green!=10"},
        ),
    ],
)
def test_cgt_feature(
    cgt_feature: CommonGroundTracking, prop, move, expected_ebank, expected_fbank
):
    """
    Test that common ground tracking works correctly.
    """
    output = cgt_feature.get_output(
        MoveInterface(speaker_id="test", move=move),
        PropositionInterface(speaker_id="test", prop=prop),
    )
    assert isinstance(output, CommonGroundInterface)

    assert output.ebank == expected_ebank
    assert output.fbank == expected_fbank
