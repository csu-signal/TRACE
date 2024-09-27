"""
Constants to set before installing the mmdemo-azure-kinect
library. These tell the compiler where to find necessary
libraries when building and tell the program where to find
necessary dll files when running.
"""

from pathlib import Path

# change these to your installation of the Azure Kinect SDK
K4A_DIR = Path(r"C:\Program Files\Azure Kinect SDK v1.4.2")
K4ABT_DIR = Path(r"C:\Program Files\Azure Kinect Body Tracking SDK")
K4A_DLL_DIR = K4ABT_DIR / "tools"
