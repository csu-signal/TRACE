# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import platform
from pathlib import Path
from glob import glob

assert platform.system() == "Windows", "This package only works on Windows"


K4A_DIR = Path("C:\\") / "Program Files" / "Azure Kinect SDK v1.4.2"
K4ABT_DIR = Path("C:\\") / "Program Files" / "Azure Kinect Body Tracking SDK"
JSON_DIR = Path(__file__).parent.parent

ext_modules = [
    Pybind11Extension(
        "azure_kinect",
        sorted(glob("src/*.cpp")),
        include_dirs=[
            str(K4A_DIR / "sdk" / "include"),
            str(K4ABT_DIR / "sdk" / "include"),
            str(JSON_DIR)
        ],
        library_dirs=[
            str(K4A_DIR / "sdk" / "windows-desktop" / "amd64" / "release" / "lib"),
            str(K4ABT_DIR / "sdk" / "windows-desktop" / "amd64" / "release" / "lib")
        ],
        libraries = ["k4a", "k4arecord", "k4abt"]
    ),
]

setup(
    name="azure_kinect_wrapper",
    ext_modules=ext_modules,
    python_requires=">=3.9",
)
