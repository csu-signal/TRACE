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


def getRadiusPoint(rUp, rDown, vectorPoint):
    up = vectorPoint.copy()
    down = vectorPoint.copy()
    up[0][1] += rUp
    down[0][1] -= rDown
    return up, down
