"""
There are 3 coordinate systems which are important for the current premade features, and this module provides helper functions for converting between them.

pixel (c, r) -- this is the 2d coordinate system of the pixels. c is the pixel column and r is the pixel row. The color and depth images use this coordinate system but need to be indexed as [r, c].

camera 3d (x_c, y_c, z_c) -- this is the 3d coordinate that the camera sees. The orientation of this system depends on the positioning and rotation of the camera. This is the coordinate system of 3d points from the old repo using Hannah's helper functions.

world 3d (x_w, y_w, z_w) -- this is a 3d coordinate system that should not change when the camera is repositioned. These values are returned by azure kinect body tracking.
"""

import cv2 as cv
import numpy as np

from mmdemo.interfaces import CameraCalibrationInterface, DepthImageInterface


def pixel_to_camera_3d(
    pixel, depth: DepthImageInterface, calibration: CameraCalibrationInterface
):
    return _convertTo3D(
        calibration.camera_matrix,
        calibration.distortion,
        depth.frame,
        int(pixel[0]),
        int(pixel[1]),
    )


def camera_3d_to_pixel(point, calibration: CameraCalibrationInterface):
    return _convert2D(point, calibration.camera_matrix, calibration.distortion)


def world_3d_to_camera_3d(point, calibration: CameraCalibrationInterface):
    return np.dot(calibration.rotation, point) + calibration.translation


def camera_3d_to_world_3d(point, calibration: CameraCalibrationInterface):
    return np.dot(np.linalg.inv(calibration.rotation), point - calibration.translation)


def _convert2D(point3D, cameraMatrix, dist):
    """
    3d camera coords to 2d pixel coords
    """
    point, _ = cv.projectPoints(
        np.array(point3D),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        cameraMatrix,
        dist,
    )

    return point[0][0]


def _convertTo3D(cameraMatrix, dist, depth, u, v):
    """
    2d (x, y) coords to 3d camera coords
    """
    z = depth[v, u]

    if z == 0:
        # print("Invalid Depth, Z returned 0")
        raise ValueError("Invalid Depth, Z returned 0")

    f_x = cameraMatrix[0, 0]
    f_y = cameraMatrix[1, 1]
    c_x = cameraMatrix[0, 2]
    c_y = cameraMatrix[1, 2]

    points_undistorted = cv.undistortPoints(
        np.array([u, v], dtype=np.float32), cameraMatrix, dist, P=cameraMatrix
    )
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    result = []
    for idx in range(points_undistorted.shape[0]):
        x = (points_undistorted[idx, 0] - c_x) / f_x * z
        y = (points_undistorted[idx, 1] - c_y) / f_y * z
        result.append(x.astype(float))
        result.append(y.astype(float))
        result.append(z.astype(float))

    return np.array(result)
