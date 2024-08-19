import cv2
import numpy as np

from mmdemo.interfaces import CameraCalibrationInterface, DepthImageInterface
from mmdemo.utils.coordinates import pixel_to_camera_3d
from mmdemo.utils.support_utils import ParseResult


def getDirectionalVector(terminal, initial):
    vectorX = terminal[0] - initial[0]
    vectorY = terminal[1] - initial[1]
    vectorZ = terminal[2] - initial[2]
    return np.array([vectorX, vectorY, vectorZ], dtype=int)


def getDirectionalVector2D(terminal, initial):
    vectorX = terminal[0] - initial[0]
    vectorY = terminal[1] - initial[1]
    return (vectorX, vectorY)


def getVectorPoint(terminal, vector):
    return (terminal[0] + vector[0], terminal[1] + vector[1], terminal[2] + vector[2])


def processPoint(
    landmarks, calibration: CameraCalibrationInterface, depth: DepthImageInterface
):
    try:
        for index, lm in enumerate(landmarks):
            if index == 5:
                bx, by = lm[0], lm[1]
            if index == 8:
                tx, ty = lm[0], lm[1]

        tip3D = pixel_to_camera_3d([int(tx), int(ty)], depth, calibration)
        base3D = pixel_to_camera_3d([int(bx), int(by)], depth, calibration)

        vector3D = getDirectionalVector(tip3D, base3D)
        nextPoint = getVectorPoint(tip3D, vector3D)
        nextPoint = getVectorPoint(nextPoint, vector3D)
        i = 1
        while i < 3:
            nextPoint = getVectorPoint(nextPoint, vector3D)
            i += 1

        # distance = distance3D(base3D, nextPoint)
        return (tx, ty, tip3D, bx, by, base3D, nextPoint, ParseResult.Success)
    except Exception as error:
        print(error)
        return (0, 0, 0, 0, 0, 0, 0, ParseResult.InvalidDepth)


def getRadiusPoint(rUp, rDown, vectorPoint):
    up = vectorPoint.copy()
    down = vectorPoint.copy()
    up[0][1] += rUp
    down[0][1] -= rDown
    return up, down
