"""
There are 3 coordinate systems which are important for the current premade features, and this module provides helper functions for converting between them.

pixel (c, r) -- this is the 2d coordinate system of the pixels. c is the pixel column and r is the pixel row. The color and depth images use this coordinate system but need to be indexed as [r, c].

camera 3d (x_c, y_c, z_c) -- this is the 3d coordinate that the camera sees. The orientation of this system depends on the positioning and rotation of the camera. This is the coordinate system of 3d points from the old repo using Hannah's helper functions.

world 3d (x_w, y_w, z_w) -- this is a 3d coordinate system that should not change when the camera is repositioned. These values are returned by azure kinect body tracking.
"""

import cv2 as cv
import numpy as np

from mmdemo.interfaces import CameraCalibrationInterface, DepthImageInterface


class CoordinateConversionError(Exception):
    pass


def pixel_to_camera_3d(
    pixel, depth: DepthImageInterface, calibration: CameraCalibrationInterface
):
    """
    2d pixel coords to 3d camera coords
    """
    try:
        z = depth.frame[int(pixel[1]), int(pixel[0])]
    except:
        z = depth.frame[int(pixel[1])-1, int(pixel[0])-1]

        print("*******************************")
        print(depth.frame.shape)
        print(int(pixel[1]), int(pixel[0]))
    # z = depth.frame[int(pixel[1]), int(pixel[0])]

    if z == 0:
        # print("Invalid Depth, Z returned 0")
        raise CoordinateConversionError("Invalid Depth, Z returned 0")

    f_x = calibration.camera_matrix[0, 0]
    f_y = calibration.camera_matrix[1, 1]
    c_x = calibration.camera_matrix[0, 2]
    c_y = calibration.camera_matrix[1, 2]

    points_undistorted = cv.undistortPoints(
        np.array(pixel, dtype=np.float32),
        calibration.camera_matrix,
        calibration.distortion,
        P=calibration.camera_matrix,
    )
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    return np.array(
        [
            (points_undistorted[0, 0] - c_x) / f_x * z,
            (points_undistorted[0, 1] - c_y) / f_y * z,
            z,
        ]
    )


def camera_3d_to_pixel(point, calibration: CameraCalibrationInterface):
    """
    3d camera coords to 2d pixel coords
    """

    point, _ = cv.projectPoints(
        np.array(point),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        calibration.camera_matrix,
        calibration.distortion,
    )
    return point[0][0].round().astype(int)


def world_3d_to_camera_3d(point, calibration: CameraCalibrationInterface):
    return np.dot(calibration.rotation, point) + calibration.translation


def camera_3d_to_world_3d(point, calibration: CameraCalibrationInterface):
    return np.dot(np.linalg.inv(calibration.rotation), point - calibration.translation)
