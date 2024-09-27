import numpy as np
import pytest

from mmdemo.features.objects.object_feature import Object
from mmdemo.features.wtd_ablation_testing.gesture_feature import (
    GestureSelectedObjectsGroundTruth,
)
from mmdemo.features.wtd_ablation_testing.object_feature import ObjectGroundTruth
from mmdemo.interfaces import (
    ColorImageInterface,
    DepthImageInterface,
    ObjectInterface3D,
    SelectedObjectsInterface,
)
from mmdemo.interfaces.data import GamrTarget
from tests.utils.features import FakeFeature


@pytest.fixture
def selected_gt(test_data_dir):
    obj_gt = ObjectGroundTruth(
        FakeFeature(),
        FakeFeature(),
        csv_path=test_data_dir / "example_ground_truth_wtd" / "objects.csv",
    )
    obj_gt.initialize()
    yield obj_gt
    obj_gt.finalize()


gt = GamrTarget


@pytest.mark.parametrize(
    "frame_counts,expected_objects",
    [
        (
            [1, 11, 21, 31, 41],
            [
                [gt.RED_BLOCK],
                [gt.PURPLE_BLOCK],
                [gt.RED_BLOCK, gt.BLUE_BLOCK],
                [gt.RED_BLOCK, gt.BLUE_BLOCK, gt.PURPLE_BLOCK],
                [gt.YELLOW_BLOCK],
            ],
        ),
        (
            [1, 2, 3, 15, 16, 21, 31, 41, 42],
            [
                [gt.RED_BLOCK],
                [],
                [],
                [gt.PURPLE_BLOCK],
                [],
                [gt.RED_BLOCK, gt.BLUE_BLOCK],
                [gt.RED_BLOCK, gt.BLUE_BLOCK, gt.PURPLE_BLOCK],
                [gt.YELLOW_BLOCK],
                [],
            ],
        ),
        (
            [50, 50, 50, 50],
            [[gt.YELLOW_BLOCK], [], [], []],
        ),
    ],
)
def test_ground_truth_gestures(
    selected_gt: ObjectGroundTruth, frame_counts, expected_objects, camera_calibration
):
    """
    Make sure that reading ground truth objects from a file
    works correctly.

    Arguments:
    `frame_counts` -- image frame counts to evaluate on
    `expected_objects` -- lists of expected GamrTargets
    """

    for frame_count, expected in zip(frame_counts, expected_objects):
        depth_interface = DepthImageInterface(
            frame=np.ones((500, 500)), frame_count=frame_count
        )
        output = selected_gt.get_output(depth_interface, camera_calibration)
        assert isinstance(output, ObjectInterface3D)
        assert len(output.objects) == len(
            expected
        ), "The wrong number of selected objects was returned"
        assert set(i.object_class for i in output.objects) == set(
            expected
        ), "The wrong objects were selected"
