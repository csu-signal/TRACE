"""
Create color, depth, and body tracking features from Azure Kinect
"""

import os

import azure_kinect_config as config

os.add_dll_directory(str(config.K4A_DLL_DIR))

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import final

import numpy as np
from _azure_kinect import Camera, Playback

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import (
    BodyTrackingInterface,
    ColorImageInterface,
    DepthImageInterface,
)


class DeviceType(Enum):
    CAMERA = auto()
    PLAYBACK = auto()


@dataclass
class _AzureKinectInterface(BaseInterface):
    """
    Store output of Azure Kinect wrapper Device class. This is a private
    interface which is just used to pass information to the color, depth,
    and body tracking features.
    """

    color: np.ndarray
    depth: np.ndarray
    body_tracking: dict
    frame_count: int


@final
class _AzureKinectDevice(BaseFeature):
    """
    Get output from Azure Kinect wrapper Device class. This is a private
    feature which is just used as input to the Azure Kinect color, depth,
    and body tracking features.
    """

    def __init__(
        self,
        *,
        device_type: DeviceType,
        camera_index: int | None = None,
        mkv_path: str | Path | None = None
    ):
        super().__init__()
        self.device_type = device_type
        self.camera_index = camera_index
        self.mkv_path = mkv_path

    def initialize(self):
        if self.device_type == DeviceType.CAMERA:
            assert self.camera_index is not None
            self.device = Camera(self.camera_index)
        elif self.device_type == DeviceType.PLAYBACK:
            assert self.mkv_path is not None
            self.device = Playback(str(self.mkv_path))

    def finalize(self):
        self.device.close()

    def get_output(self):
        frame_count = self.device.get_frame_count()
        color, depth, body_tracking = self.device.get_frame()
        if color is None or depth is None or len(body_tracking) == 0:
            return None

        return _AzureKinectInterface(color, depth, body_tracking, frame_count)


@final
class AzureKinectColor(BaseFeature):
    """
    Feature to get color images from Azure Kinect.

    The input interface is `_AzureKinectInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `ColorImageInterface`.
    """

    def get_output(
        self, azure_input: _AzureKinectInterface
    ) -> ColorImageInterface | None:
        if not azure_input.is_new():
            return None
        return ColorImageInterface(
            frame_count=azure_input.frame_count, frame=azure_input.color
        )


@final
class AzureKinectDepth(BaseFeature):
    """
    Feature to get depth images from Azure Kinect.

    The input interface is `_AzureKinectInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `DepthImageInterface`.
    """

    def get_output(
        self, azure_input: _AzureKinectInterface
    ) -> DepthImageInterface | None:
        if not azure_input.is_new():
            return None
        return DepthImageInterface(
            frame_count=azure_input.frame_count, frame=azure_input.depth
        )


@final
class AzureKinectBodyTracking(BaseFeature):
    """
    Feature to get body tracking info from Azure Kinect.

    The input interface is `_AzureKinectInterface`, which is a private
    interface that is created using helper functions.

    The output interface is `BodyTrackingInterface`.
    """

    def get_output(
        self, azure_input: _AzureKinectInterface
    ) -> BodyTrackingInterface | None:
        if not azure_input.is_new():
            return None
        return BodyTrackingInterface(
            bodies=azure_input.body_tracking["bodies"],
            timestamp_usec=azure_input.body_tracking["timestamp_usec"],
        )


def create_azure_kinect_features(
    device_type: DeviceType,
    *,
    camera_index: int | None = None,
    mkv_path: str | Path | None = None
):
    """
    Returns 3 features which output ColorImageInterface, DepthImageInterface,
    and BodyTrackingInterface using information from an Azure Kinect camera or
    playback mkv file.

    Arguments:
    device_type -- an instance of the DeviceType enumeration specifying if the
    data should come from a camera or playback

    Keyword Arguments:
    camera_index -- the index of the Azure Kinect camera. Only required when
                    `device_type == DeviceType.CAMERA`
    mkv_path -- the path to an Azure Kinect playback mkv file. Only required
                when `device_type == DeviceType.PLAYBACK`
    """
    input_feature = _AzureKinectDevice(
        device_type=device_type, camera_index=camera_index, mkv_path=mkv_path
    )

    color = AzureKinectColor(input_feature)
    depth = AzureKinectDepth(input_feature)
    body_tracking = AzureKinectBodyTracking(input_feature)

    return color, depth, body_tracking
