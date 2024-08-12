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
    of dependencies. It will assert that at least one new input is received,
    and will check a few basic properties about the structure of input
    interfaces.
    """

    def __init__(self, *args, max_frames=None):
        super().__init__(*args)
        self.max_frames = max_frames

    def initialize(self):
        self.got_new_data = False
        self.internal_count = 0

    def finalize(self):
        assert self.got_new_data, "No new input was ever received"

    def is_done(self) -> bool:
        return self.max_frames != None and self.internal_count > self.max_frames

    def get_output(self, *args: BaseInterface):
        self.internal_count += 1

        for interface in args:
            if not interface.is_new():
                continue

            self.got_new_data = True

            if isinstance(interface, ColorImageInterface):
                assert isinstance(interface.frame, np.ndarray)
                assert len(interface.frame.shape) == 3
                assert interface.frame.shape[2] == 3

            if isinstance(interface, DepthImageInterface):
                assert isinstance(interface.frame, np.ndarray)
                assert len(interface.frame.shape) == 2

            if isinstance(interface, BodyTrackingInterface):
                assert hasattr(interface, "bodies")

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
    Check that loading from mkv works
    """
    from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

    color, depth, body_tracking = create_azure_kinect_features(
        device_type=DeviceType.PLAYBACK, mkv_path=mkv_path, playback_end_seconds=2
    )

    Demo(targets=[CheckOutput(i) for i in (color, depth, body_tracking)]).run()


@pytest.mark.xfail
def test_camera():
    """
    Check that camera output works
    """
    from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

    color, depth, body_tracking = create_azure_kinect_features(
        device_type=DeviceType.CAMERA, camera_index=0
    )

    Demo(
        targets=[CheckOutput(i, max_frames=10) for i in (color, depth, body_tracking)]
    ).run()
