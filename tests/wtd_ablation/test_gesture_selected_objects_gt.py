import numpy as np
import pytest

from mmdemo.features.wtd_ablation_testing.gesture_feature import (
    GestureSelectedObjectsGroundTruth,
)
from mmdemo.interfaces import ColorImageInterface, SelectedObjectsInterface
from mmdemo.interfaces.data import GamrTarget
from tests.utils.features import FakeFeature


@pytest.fixture
def selected_gt(test_data_dir):
    sel_gt = GestureSelectedObjectsGroundTruth(
        FakeFeature(),
        csv_path=test_data_dir / "example_ground_truth_wtd" / "gestures.csv",
    )
    sel_gt.initialize()
    yield sel_gt
    sel_gt.finalize()


gt = GamrTarget


@pytest.mark.parametrize(
    "frame_counts,expected_selections",
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
    selected_gt: GestureSelectedObjectsGroundTruth, frame_counts, expected_selections
):
    """
    Make sure that reading ground truth gestures from a file
    works correctly. These are really the ground truth objects
    which have been selected by gesture.

    Arguments:
    `frame_counts` -- image frame counts to evaluate on
    `expected_selections` -- lists of expected GamrTargets
    """

    for frame_count, expected in zip(frame_counts, expected_selections):
        color_interface = ColorImageInterface(
            frame=np.zeros((5, 5, 3)), frame_count=frame_count
        )
        output = selected_gt.get_output(color_interface)
        assert isinstance(output, SelectedObjectsInterface)
        assert len(output.objects) == len(
            expected
        ), "The wrong number of selected objects was returned"
        assert set(i[0].object_class for i in output.objects) == set(
            expected
        ), "The wrong objects were selected"
