import pytest

from mmdemo.features.dense_paraphrasing.dense_paraphrasing_feature import (
    DenseParaphrasing,
)
from mmdemo.features.dense_paraphrasing.frame_time_converter import FrameTimeConverter
from mmdemo.features.dense_paraphrasing.referenced_objects_feature import (
    ReferencedObjects,
)
from mmdemo.interfaces import (
    EmptyInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
)
from mmdemo.interfaces.data import GamrTarget, ObjectInfo2D
from tests.utils.fake_feature import FakeFeature


@pytest.fixture
def referenced_objects(monkeypatch):
    ref = ReferencedObjects(FakeFeature(), FakeFeature())
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

    # override the frame time converter such that frames
    # and times are equal
    monkeypatch.setattr(ref, "frame_time_converter", FakeFrameTimeConverter())

    # override the frame bins such that a new frame
    # bin is created every frame
    def override_frame_bin(frame):
        return frame

    monkeypatch.setattr(ref, "get_frame_bin", override_frame_bin)

    yield ref
    ref.finalize()


@pytest.mark.parametrize(
    "data,ft_expected,tf_expected",
    [
        (
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            [(0, 0), (2, 2), (4, 4)],
            [(0, 0), (3, 3), (4, 4)],
        ),
        (
            [(0, 5), (2, 6), (8, 20), (13, 21), (19, 22)],
            [(0, 5), (19, 22), (7, 6), (10, 20), (20, 22)],
            [(5, 0), (22, 19), (10, 2), (50, 19)],
        ),
    ],
)
def test_frame_time_converter(data, ft_expected, tf_expected):
    converter = FrameTimeConverter()
    for frame, time in data:
        converter.add_data(frame, time)

    for f, t in ft_expected:
        assert converter.get_time(f) == t

    for t, f in tf_expected:
        assert converter.get_frame(t) == f


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
    referenced_objects: ReferencedObjects,
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
