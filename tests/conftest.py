from pathlib import Path

import pytest

from tests.utils.data import read_frame_pkl, read_point_cloud_pkl


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(
    params=[
        "gesture_01.pkl",
        "gesture_02.pkl",
        "frame_01.pkl",
        "frame_02.pkl",
    ]
)
def azure_kinect_frame_file(request, test_data_dir):
    """
    Files for azure kinect frames
    """
    file = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    return file


@pytest.fixture
def azure_kinect_frame(azure_kinect_frame_file):
    """
    Returns (color, depth, body tracking, calibration)
    interfaces
    """
    return read_frame_pkl(azure_kinect_frame_file)


@pytest.fixture
def camera_calibration(azure_kinect_frame_file):
    """
    Return camera calibration interface
    """
    _, _, _, cal = read_frame_pkl(azure_kinect_frame_file)
    return cal


@pytest.fixture(params=["point_cloud_01.pkl"])
def point_cloud(request, test_data_dir):
    """
    Return (3d point map, depth, calibration) for testing.

    The 3d point map is a (h,w,3) numpy vector where
    the first two coords correspond to the pixel and
    the last coord corresponds to the (x,y,z) position
    in 3d space of that pixle.
    """
    file = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    cloud, depth, cal = read_point_cloud_pkl(file)
    return cloud, depth, cal
