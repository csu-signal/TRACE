import pytest

from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface
from tests.utils.fake_feature import FakeFeature


@pytest.fixture(scope="module")
def proposition_feature():
    prop = Proposition(FakeFeature())
    prop.initialize()
    yield prop
    prop.finalize()


@pytest.mark.model_dependent
@pytest.mark.parametrize(
    "text,expected",
    [
        ("red is 10", "red = 10"),
        ("10", "no prop"),
        ("red", "no prop"),
        ("green is 20", "green = 20"),
        ("red weighs more than blue", "red > blue"),
        ("red is the same as blue", "blue = red"),
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

    output_set = set([i.strip() for i in output.prop.split(",")])
    expected_set = set([i.strip() for i in expected.split(",")])

    assert output_set == expected_set
    assert output.speaker_id == "test"
