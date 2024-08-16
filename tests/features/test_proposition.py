import pytest

from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface


@pytest.fixture(scope="module")
def proposition_feature():
    prop = Proposition()
    prop.initialize()
    yield prop
    prop.finalize()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("red is 10", "red = 10"),
        ("green is 20", "green = 20"),
        ("red is more than blue", "red > blue"),
    ],
)
def test_prop_extraction(
    proposition_feature: Proposition,
    text,
    expected,
):
    """
    Test that prop extraction works correctly.
    """
    output = proposition_feature.get_output(
        TranscriptionInterface(speaker_id="test", start_time=0, end_time=1, text=text)
    )
    assert isinstance(output, PropositionInterface)

    assert output.prop == expected
    assert output.speaker_id == "test"
