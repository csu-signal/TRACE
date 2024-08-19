"""
Test coordinate conversions

Right now correctness is checked by performing a conversion and then the
inverse of its conversion. More tests should be added later with specific
calibration matrix values.
"""

import numpy as np
import pytest

from mmdemo.utils.coordinates import (
    camera_3d_to_pixel,
    camera_3d_to_world_3d,
    pixel_to_camera_3d,
    world_3d_to_camera_3d,
)
from tests.utils.data import read_frame_pkl


@pytest.fixture(params=["frame_01.pkl"])
def test_frame_depth_and_calibration(request, test_data_dir):
    file = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    _, depth, _, cal = read_frame_pkl(file)
    return depth, cal


@pytest.fixture(params=[(100, 100, 100), (100, 50, 150), (340, 200, 100)])
def point_3d(request):
    return np.array(request.param, dtype=np.float32)


@pytest.fixture(params=[(500, 500), (700, 900), (1000, 500)])
def point_2d(request):
    return np.array(request.param, dtype=np.int32)


@pytest.mark.xfail(reason="I think we need points on the depth image for this to work")
def test_camera_3d_inverse(point_3d, test_frame_depth_and_calibration):
    depth, calibration = test_frame_depth_and_calibration

    output = camera_3d_to_pixel(point_3d, calibration)
    output = pixel_to_camera_3d(output, depth, calibration)

    assert np.allclose(point_3d, output)
    assert point_3d.dtype == output.dtype


def test_2d_inverse(point_2d, test_frame_depth_and_calibration):
    depth, calibration = test_frame_depth_and_calibration

    output = pixel_to_camera_3d(point_2d, depth, calibration)
    output = camera_3d_to_pixel(output, calibration)

    assert np.allclose(point_2d, output)
    assert point_2d.dtype == output.dtype


def test_camera_world_inverse(point_3d, test_frame_depth_and_calibration):
    _, calibration = test_frame_depth_and_calibration

    output = camera_3d_to_world_3d(point_3d, calibration)
    output = world_3d_to_camera_3d(output, calibration)

    assert np.allclose(point_3d, output)
    assert point_3d.dtype == output.dtype
