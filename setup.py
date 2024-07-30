from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import platform
from pathlib import Path
from glob import glob
import os
import urllib.request

# change these to your installation of the Azure Kinect SDK
K4A_DIR = Path(r"C:\Program Files\Azure Kinect SDK v1.4.2")
K4ABT_DIR = Path(r"C:\Program Files\Azure Kinect Body Tracking SDK")


assert platform.system() == "Windows", "This package only works on Windows"

# download json.hpp if needed
json_dir = Path(".").resolve()
if not os.path.exists(json_dir / "nlohmann/json.hpp"):
    os.makedirs(json_dir / "nlohmann", exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp",
        json_dir / "nlohmann/json.hpp",
    )

# create extension for azure kinect using pybind
ext_modules = [
    Pybind11Extension(
        "azure_kinect",
        sorted(glob("azure_kinect_wrapper/*.cpp")),
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
    name="azure_kinect_wrapper",
    python_requires=">=3.10",
    ext_modules=ext_modules,
    packages=["azure_kinect-stubs", "demo"],
    include_package_data=True,
    package_data={"azure_kinect-stubs": ["*.pyi"]}
)
