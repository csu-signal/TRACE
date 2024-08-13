"""
Features which can be used as dependencies in a demo
"""
# TODO: add camera calibration data

from pathlib import Path
from typing import final

from mmdemo_azure_kinect.azure_kinect_output import (
    _AzureKinectDevice,
    _AzureKinectInterface,
)
from mmdemo_azure_kinect.device_type import DeviceType

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    BodyTrackingInterface,
    ColorImageInterface,
    DepthImageInterface,
)


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
    mkv_path: str | Path | None = None,
    mkv_frame_rate: int | None = 30,
    playback_frame_rate: int | None = 5,
    playback_end_seconds: int | None = None
):
    """
    Returns 3 features which output ColorImageInterface, DepthImageInterface,
    and BodyTrackingInterface using information from an Azure Kinect camera or
    playback mkv file.

    Arguments:
    `device_type` -- an instance of the DeviceType enum specifying if the
    data should come from a camera or playback

    Keyword Arguments:
    `camera_index` -- the index of the Azure Kinect camera, used for `DeviceType.CAMERA`
    `mkv_path` -- the path to an Azure Kinect playback mkv file, used for `DeviceType.PLAYBACK`
    `mkv_frame_rate` -- frame rate of the mkv file, default 30, used for `DeviceType.PLAYBACK`
    `playback_frame_rate` -- simulated frame rate of the playback, default 5, this can reduce the number of frames which need to be processed, used for `DeviceType.PLAYBACK`
    `playback_end_seconds` -- the number of seconds to end playback after or None, default None, used for `DeviceType.PLAYBACK`
    """
    input_feature = _AzureKinectDevice(
        device_type=device_type,
        camera_index=camera_index,
        mkv_path=mkv_path,
        mkv_frame_rate=mkv_frame_rate,
        playback_frame_rate=playback_frame_rate,
        playback_end_seconds=playback_end_seconds,
    )

    color = AzureKinectColor(input_feature)
    depth = AzureKinectDepth(input_feature)
    body_tracking = AzureKinectBodyTracking(input_feature)

    return color, depth, body_tracking
