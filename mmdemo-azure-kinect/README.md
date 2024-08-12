# mmdemo-azure-kinect

This directory contains a separate Python package that allows for creating demo features from Azure Kinect cameras and mkv playback files. This will only successfully build on Windows, so it is a separate package in order to allow the main library to be used on any operating system.

## Setup Instructions

### Azure Kinect SDK

Both the [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#installation) and [Body Tracking SDK](https://learn.microsoft.com/en-us/azure/kinect-dk/body-sdk-download) are required and can be downloaded/installed for Windows from the linked websites. Use version 1.4.2 of the Azure Kinect SDK and version 1.1.2 of the body tracking SDK if possible.

Once the installation is complete, open `azure_kinect_config.py`. Ensure that `K4A_DIR` is set to the correct Azure Kinect SDK path, `K4ABT_DIR` is set to the correct Azure Kinect Body Tracking SDK path, and `K4A_DLL_DIR` is set to the directory containing DLL files for Azure Kinect.

### Install package

From the root directory of this repository, run `pip install -e .\mmdemo-azure-kinect` to build and install the package.

### Modules

These setup instructions will install the following:
- `mmdemo_azure_kinect`, a module which provides functions for creating Azure Kinect input features that can be used with the `mmdemo` library.
- `_azure_kinect`, Python bindings for the Azure Kinect SDK. These can be used to create camera/playback devices and obtain color, depth, and body tracking information.
- `_azure_kinect-stubs`, type stubs for `_azure_kinect`. This will be used by an LSP to provide type hints and docstrings for the Python bindings.

## Example Usage
```python
from mmdemo_azure_kinect import create_azure_kinect_features, DeviceType

color, depth, body_tracking = create_azure_kinect_features(
    device_type=DeviceType.CAMERA, camera_index=0
)
```
In the example above, `color`, `depth`, and `body_tracking` are features which can be used as dependencies of other features in a demo.