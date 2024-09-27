"""
Create color, depth, and body tracking features from Azure Kinect
"""

from mmdemo_azure_kinect.device_type import DeviceType
from mmdemo_azure_kinect.features import (
    AzureKinectBodyTracking,
    AzureKinectColor,
    AzureKinectDepth,
    create_azure_kinect_features,
)
