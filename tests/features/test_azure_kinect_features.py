from typing import final

import numpy as np
import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.demo import Demo
from mmdemo.interfaces import (
    BodyTrackingInterface,
    ColorImageInterface,
    DepthImageInterface,
)


@pytest.fixture
def mkv_path():
    return r"C:\Users\brady\Desktop\Group_01-master.mkv"


@final
class CheckOutput(BaseFeature):
    """
    Helper class to test azure kinect features. Can use any number
    of dependencies.
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        body_tracking: BaseFeature[BodyTrackingInterface],
        max_frames=None,
    ):
        super().__init__(color, depth, body_tracking)
        self.max_frames = max_frames

    def initialize(self):
        self.got_new_data = False
        self.internal_count = 0

    def finalize(self):
        assert self.got_new_data, "No new input was ever received"

    def is_done(self) -> bool:
        return self.max_frames != None and self.internal_count > self.max_frames

    def get_output(
        self,
        color: ColorImageInterface,
        depth: DepthImageInterface,
        body_tracking: BodyTrackingInterface,
    ):
        self.internal_count += 1

        if not color.is_new() or not depth.is_new() or not body_tracking.is_new():
            return None

        self.got_new_data = True

        # make sure types are correct
        assert isinstance(color, ColorImageInterface), "Incorrect interface for color"
        assert isinstance(color.frame, np.ndarray), "Color image must be a numpy array"
        assert len(color.frame.shape) == 3, "Color image must have dim 3"
        assert color.frame.shape[2] == 3, "Color image must have 3 color channels"

        assert isinstance(depth, DepthImageInterface), "Incorrect interface for depth"
        assert isinstance(depth.frame, np.ndarray), "Depth image must be a numpy array"
        assert len(depth.frame.shape) == 2, "depth image must have dim 2"

        assert isinstance(
            body_tracking, BodyTrackingInterface
        ), "Incorrect interface for body tracking"

        return None


@pytest.mark.xfail
def test_import():
    """
    Check that imports work
    """
    from mmdemo_azure_kinect import (
        AzureKinectBodyTracking,
        AzureKinectColor,
        AzureKinectDepth,
        DeviceType,
        create_azure_kinect_features,
    )


@pytest.mark.xfail
def test_playback(mkv_path):
    """
    Check that loading from mkv works. These features need to be run
    in a demo instead of tested manually since they are driven by a
    private feature.
    """
    from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

    color, depth, body_tracking = create_azure_kinect_features(
        device_type=DeviceType.PLAYBACK, mkv_path=mkv_path, playback_end_seconds=2
    )

    Demo(targets=[CheckOutput(color, depth, body_tracking)]).run()


@pytest.mark.xfail
def test_camera():
    """
    Check that camera output works. These features need to be run
    in a demo instead of tested manually since they are driven by a
    private feature.
    """
    from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

    color, depth, body_tracking = create_azure_kinect_features(
        device_type=DeviceType.CAMERA, camera_index=0
    )
    Demo(targets=[CheckOutput(color, depth, body_tracking, max_frames=10)]).run()
