"""
Test coordinate conversions

Right now correctness is checked by performing a conversion and then the
inverse of its conversion. More tests should be added later with specific
calibration matrix values.
"""

import numpy as np
import pytest

from mmdemo.utils.coordinates import (
    CoordinateConversionError,
    camera_3d_to_pixel,
    camera_3d_to_world_3d,
    pixel_to_camera_3d,
    world_3d_to_camera_3d,
)

# the interval at which rows and cols
# of the point cloud will be selected for testing
INTERVAL = 17


def test_camera_3d_to_pixel(point_cloud):
    cloud, depth, calibration = point_cloud

    assert (
        np.linalg.norm(depth) > 0
    ), "Depth image only contains 0 so testing will not work correctly"

    for r in range(0, cloud.shape[0], INTERVAL):
        for c in range(0, cloud.shape[1], INTERVAL):
            point_3d = cloud[r, c]
            if np.linalg.norm(point_3d) == 0:
                # don't test when 3d point is invalid
                pass

            expected = np.array([c, r])
            output = camera_3d_to_pixel(point_3d, calibration)

            assert np.allclose(output, expected)
            assert output.dtype == expected.dtype


def test_pixel_to_camera_3d(point_cloud):
    cloud, depth, calibration = point_cloud

    assert (
        np.linalg.norm(depth) > 0
    ), "Depth image only contains 0 so testing will not work correctly"

    for r in range(0, cloud.shape[0], INTERVAL):
        for c in range(0, cloud.shape[1], INTERVAL):
            point_2d = np.array([c, r])

            expected = cloud[r, c]
            try:
                output = pixel_to_camera_3d(point_2d, depth, calibration)
                assert np.allclose(output, expected)
                assert output.dtype == expected.dtype

            except CoordinateConversionError:
                assert (
                    np.linalg.norm(expected) == 0
                ), "Conversion should fail exactly when the point cloud has (0,0,0)"


@pytest.fixture(params=[(100, 200, 300), (100, 500, 90), (-10, -50, 80)])
def point_3d(request):
    return np.array(request.param, dtype=np.float64)


def test_camera_world_inverse(point_3d, camera_calibration):
    output = camera_3d_to_world_3d(point_3d, camera_calibration)
    output = world_3d_to_camera_3d(output, camera_calibration)

    assert np.allclose(point_3d, output)
    assert point_3d.dtype == output.dtype
