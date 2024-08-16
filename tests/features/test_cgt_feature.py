import pytest

from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.interfaces import CommonGroundInterface, MoveInterface, PropositionInterface


@pytest.fixture(scope="module")
def cgt_feature():
    cgt = CommonGroundTracking()
    cgt.initialize()
    yield cgt
    cgt.finalize()


@pytest.mark.parametrize(
    "prop, move, expected",
    [
        ("red = 10", "STATEMENT"),
        ("10", "STATEMENT"),
        ("red", "STATEMENT"),
        ("green = 20", "STATEMENT"),
        ("red > blue", "STATEMENT"),
        ("red = blue", "STATEMENT"),
        ("yellow != 40", "DOUBT"),
        ("red != green", "DOUBT"),
        ("I agree", "ACCEPT"),
    ],
)
def test_cgt_feature(cgt_feature: CommonGroundTracking, input, expected):
    """
    Test that common ground tracking works correctly.
    """
    output = cgt_feature.get_output()
