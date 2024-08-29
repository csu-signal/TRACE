import pytest

from mmdemo.features.objects.accumulated_selected_objects_feature import (
    AccumulatedSelectedObjects,
)
from mmdemo.interfaces import SelectedObjectsInterface, TranscriptionInterface
from mmdemo.interfaces.data import GamrTarget, ObjectInfo2D
from tests.utils.fake_feature import FakeFeature


@pytest.fixture
def referenced_objects(monkeypatch):
    ref = AccumulatedSelectedObjects(FakeFeature(), FakeFeature())
    ref.initialize()

    # fake frame time converter where the time is always
    # 5 more than the frame
    class FakeFrameTimeConverter:
        def add_data(self, frame, time):
            pass

        def get_frame(self, time):
            return time - 5

        def get_time(self, frame):
            return frame + 5

        def get_num_datapoints(self):
            return 2

    # override the frame time converter such that frames
    # and times are equal
    monkeypatch.setattr(ref, "frame_time_converter", FakeFrameTimeConverter())

    yield ref
    ref.finalize()


gt = GamrTarget


@pytest.mark.parametrize(
    "classes_by_bin,expected_classes,utterance_frames",
    [
        ([[gt.RED_BLOCK]], [gt.RED_BLOCK], (0, 0)),
        (
            [[gt.RED_BLOCK], [gt.BLUE_BLOCK]],
            [gt.RED_BLOCK, gt.BLUE_BLOCK],
            (0, 1),
        ),
        (
            [[gt.RED_BLOCK, gt.BLUE_BLOCK]],
            [gt.RED_BLOCK, gt.BLUE_BLOCK],
            (0, 0),
        ),
        (
            [
                [gt.YELLOW_BLOCK],
                [gt.RED_BLOCK],
                [gt.BLUE_BLOCK, gt.RED_BLOCK],
                [gt.YELLOW_BLOCK],
            ],
            [gt.RED_BLOCK, gt.BLUE_BLOCK],
            (1, 2),
        ),
        (
            [
                [gt.YELLOW_BLOCK],
                [],
                [gt.BLUE_BLOCK, gt.RED_BLOCK],
                [gt.YELLOW_BLOCK],
            ],
            [gt.BLUE_BLOCK, gt.RED_BLOCK],
            (1, 2),
        ),
    ],
)
def test_referenced_objects(
    referenced_objects: AccumulatedSelectedObjects,
    classes_by_bin,
    expected_classes,
    utterance_frames,
):
    """
    Make sure the correct referenced objects are returned.
    These should not include any repeats and only consider
    objects within the utterance time.
    """
    old_transcription = TranscriptionInterface("", 0, 0, "")
    old_transcription._new = False

    old_objects = SelectedObjectsInterface([])
    old_objects._new = False

    for sel_classes in classes_by_bin:
        objects = []
        objects.append(
            (
                ObjectInfo2D(
                    p1=(0, 0), p2=(1, 1), object_class=GamrTarget.PURPLE_BLOCK
                ),
                False,
            )
        )
        for cl in sel_classes:
            objects.append((ObjectInfo2D(p1=(0, 0), p2=(1, 1), object_class=cl), True))
        objects.append(
            (
                ObjectInfo2D(p1=(0, 0), p2=(1, 1), object_class=GamrTarget.GREEN_BLOCK),
                False,
            )
        )

        output = referenced_objects.get_output(
            SelectedObjectsInterface(objects), old_transcription
        )
        assert (
            output is None
        ), "there should be no referenced objects when there is no transcription"

    # start and end time are offset to make sure
    # overridden frame time converter is being used
    output = referenced_objects.get_output(
        old_objects,
        TranscriptionInterface(
            speaker_id="test",
            start_time=utterance_frames[0] + 5,
            end_time=utterance_frames[1] + 5,
            text="",
        ),
    )
    assert isinstance(output, SelectedObjectsInterface)
    assert all(
        [i[1] for i in output.objects]
    ), "All returned referenced objects should be selected"

    assert [i[0].object_class for i in output.objects] == expected_classes
