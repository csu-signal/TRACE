import pytest

from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface


@pytest.fixture(scope="module")
def proposition_feature():
    prop = Proposition()
    prop.initialize()
    yield prop
    prop.finalize()


@pytest.mark.xfail(reason="Prop extractor model does not work for many test cases")
@pytest.mark.parametrize(
    "text,expected",
    [
        ("red is 10", "red = 10"),
        ("10", "no prop"),
        ("red", "no prop"),
        ("green is 20", "green = 20"),
        ("red weighs more than blue", "red > blue"),
        ("red is the same as blue", "red = blue"),
        ("there should be no prop here", "no prop"),
        ("yellow is more than 30", "yellow > 30"),
        ("yellow weighs more than 30", "yellow > 30"),
        ("purple weighs less than 50", "purple < 50"),
        ("red is 10 and yellow is 20", "red = 10, yellow = 20"),
        ("10 20 30 40 50", "no prop"),
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
