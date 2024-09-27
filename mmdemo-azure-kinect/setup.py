import os
import platform
import sys
import urllib.request
from glob import glob
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

sys.path.append(str(Path(__file__).parent))
from azure_kinect_config import K4A_DIR, K4ABT_DIR

assert platform.system() == "Windows", "This package only works on Windows"

# download json.hpp if needed
json_dir = Path(".")
if not os.path.exists(json_dir / "nlohmann/json.hpp"):
    os.makedirs(json_dir / "nlohmann", exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp",
        json_dir / "nlohmann/json.hpp",
    )

# create extension for azure kinect using pybind
ext_modules = [
    Pybind11Extension(
        "_azure_kinect",
        sorted(glob("src/*.cpp")),
        include_dirs=[
            str(K4A_DIR / "sdk/include"),
            str(K4ABT_DIR / "sdk/include"),
            str(json_dir),
        ],
        library_dirs=[
            str(K4A_DIR / "sdk/windows-desktop/amd64/release/lib"),
            str(K4ABT_DIR / "sdk/windows-desktop/amd64/release/lib"),
            str(K4ABT_DIR / "tools"),
        ],
        libraries=["k4a", "k4arecord", "k4abt"],
    ),
]

# setup package
setup(
    name="mmdemo-azure-kinect",
    python_requires=">=3.10",
    ext_modules=ext_modules,
    packages=["mmdemo_azure_kinect", "_azure_kinect-stubs"],
    py_modules=["azure_kinect_config"],
    include_package_data=True,
    package_data={"_azure_kinect-stubs": ["*.pyi"]},
)
