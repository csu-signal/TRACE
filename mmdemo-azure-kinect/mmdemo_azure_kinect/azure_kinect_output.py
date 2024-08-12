"""
Private interface and feature to get output of Azure Kinect
"""

import os

import azure_kinect_config as config

# dll's are needed to import _azure_kinect
os.add_dll_directory(str(config.K4A_DLL_DIR))

# the dll directory also needs to be on PATH
# because the body tracking SDK searches for
# them
os.environ["PATH"] += ";" + str(config.K4A_DLL_DIR)

from dataclasses import dataclass
from pathlib import Path
from typing import final

import cv2 as cv
import numpy as np
from _azure_kinect import Camera, Playback
from mmdemo_azure_kinect.device_type import DeviceType

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface


@dataclass
class _AzureKinectInterface(BaseInterface):
    """
    Store output of Azure Kinect wrapper Device class. This is a private
    interface which is just used to pass information to the color, depth,
    and body tracking features.

    color -- color image in bgra
    depth -- depth image
    body_tracking -- body tracking output dict
    frame_count -- current frame
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
        camera_index: int | None,
        mkv_path: str | Path | None,
        mkv_frame_rate: int | None,
        playback_frame_rate: int | None,
        playback_end_seconds: int | None
    ):
        """
        See `mmdemo_azure_kinect.create_azure_kinect_features` for argument info
        """
        super().__init__()
        self.device_type = device_type
        self.camera_index = camera_index
        self.mkv_path = mkv_path
        self.mkv_frame_rate = mkv_frame_rate
        self.playback_frame_rate = playback_frame_rate
        self.playback_end_seconds = playback_end_seconds

    def initialize(self):
        if self.device_type == DeviceType.CAMERA:
            assert self.camera_index is not None, "Camera index required"
            self.device = Camera(self.camera_index)
        elif self.device_type == DeviceType.PLAYBACK:
            assert self.mkv_path is not None, "MKV path required"
            assert self.mkv_frame_rate is not None, "MKV frame rate required"
            assert self.playback_frame_rate is not None, "Playback frame rate required"
            assert (
                self.mkv_frame_rate / self.playback_frame_rate
                == self.mkv_frame_rate // self.playback_frame_rate
            ), "Playback frame rate must be a divisor of mkv frame rate"
            self.device = Playback(str(self.mkv_path))

        self.frame_count = 0

    def finalize(self):
        self.device.close()

    def get_output(self):
        if self.device_type == DeviceType.PLAYBACK:
            self.device.skip_frames(  # pyright: ignore
                self.mkv_frame_rate // self.playback_frame_rate - 1  # pyright: ignore
            )

        self.frame_count = self.device.get_frame_count()
        color, depth, body_tracking = self.device.get_frame()
        if color is None or depth is None or len(body_tracking) == 0:
            return None

        color_rgb = cv.cvtColor(color, cv.COLOR_BGRA2RGB)

        return _AzureKinectInterface(
            color_rgb,
            depth,
            body_tracking,
            self.frame_count,
        )

    def is_done(self) -> bool:
        if (
            self.device_type == DeviceType.PLAYBACK
            and self.playback_end_seconds is not None
            and self.frame_count
            > self.mkv_frame_rate * self.playback_end_seconds  # pyright: ignore
        ):
            return True

        return False
