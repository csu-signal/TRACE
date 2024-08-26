import pytest

from mmdemo.features.transcription.dense_paraphrasing_feature import (
    DenseParaphrasedTranscription,
)
from mmdemo.interfaces import SelectedObjectsInterface, TranscriptionInterface
from mmdemo.interfaces.data import GamrTarget, ObjectInfo2D
from tests.utils.fake_feature import FakeFeature


@pytest.fixture(scope="module")
def dense_paraphrasing_feature():
    dp = DenseParaphrasedTranscription(FakeFeature(), FakeFeature())
    dp.initialize()
    yield dp
    dp.finalize()


@pytest.mark.parametrize(
    "original_text,object_classes,expected",
    [
        ("this one is 10", [GamrTarget.RED_BLOCK], "red one is 10"),
        ("This one is 10", [GamrTarget.RED_BLOCK], "red one is 10"),
        (
            "This is 10 and that is 20",
            [GamrTarget.RED_BLOCK, GamrTarget.BLUE_BLOCK],
            "red is 10 and blue is 20",
        ),
        (
            "These are both 10",
            [GamrTarget.RED_BLOCK, GamrTarget.BLUE_BLOCK],
            "red, blue are both 10",
        ),
        (
            "This is 20 and those are 10",
            [GamrTarget.GREEN_BLOCK, GamrTarget.RED_BLOCK, GamrTarget.BLUE_BLOCK],
            "green is 20 and green, red, blue are 10",
        ),
        (
            "This is 20",
            [],
            "This is 20",
        ),
        (
            "These are 20",
            [],
            "These are 20",
        ),
        (
            "This is 10 and that is 20",
            [GamrTarget.RED_BLOCK],
            "red is 10 and that is 20",
        ),
    ],
)
def test_dense_paraphrasing(
    dense_paraphrasing_feature,
    original_text,
    object_classes,
    expected,
):
    # create objects list of both selected and unselected objects
    objects = [
        (
            ObjectInfo2D(
                p1=(0, 0),
                p2=(1, 1),
                object_class=GamrTarget.YELLOW_BLOCK,
            ),
            False,
        )
    ]
    for i in object_classes:
        objects.append((ObjectInfo2D(p1=(0, 0), p2=(1, 1), object_class=i), True))
    objects.append(
        (
            ObjectInfo2D(
                p1=(0, 0),
                p2=(1, 1),
                object_class=GamrTarget.YELLOW_BLOCK,
            ),
            False,
        )
    )

    output = dense_paraphrasing_feature.get_output(
        TranscriptionInterface("test", 13, 15, original_text),
        SelectedObjectsInterface(objects=objects),
    )
    assert isinstance(output, TranscriptionInterface)

    assert output.speaker_id == "test"
    assert output.start_time == 13
    assert output.end_time == 15

    assert output.text == expected
