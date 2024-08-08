import copy
import itertools
import json
import math
import os
from enum import Enum
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from numpy.linalg import norm


# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
class Handedness(Enum):
    Left = "Right"
    Right = "Left"


class ParseResult(Enum):
    Unknown = (0,)
    Success = (1,)
    Exception = (2,)
    InvalidDepth = (3,)
    NoObjects = (4,)
    NoGamr = 5


class Joint(Enum):
    PELVIS = 0
    SPINE_NAVEL = 1
    SPINE_CHEST = 2
    NECK = 3
    CLAVICLE_LEFT = 4
    SHOULDER_LEFT = 5
    ELBOW_LEFT = 6
    WRIST_LEFT = 7
    HAND_LEFT = 8
    HANDTIP_LEFT = 9
    THUMB_LEFT = 10
    CLAVICLE_RIGHT = 11
    SHOULDER_RIGHT = 12
    ELBOW_RIGHT = 13
    WRIST_RIGHT = 14
    HAND_RIGHT = 15
    HANDTIP_RIGHT = 16
    THUMB_RIGHT = 17
    HIP_LEFT = 18
    KNEE_LEFT = 19
    ANKLE_LEFT = 20
    FOOT_LEFT = 21
    HIP_RIGHT = 22
    KNEE_RIGHT = 23
    ANKLE_RIGHT = 24
    FOOT_RIGHT = 25
    HEAD = 26
    NOSE = 27
    EYE_LEFT = 28
    EAR_LEFT = 29
    EYE_RIGHT = 30
    EAR_RIGHT = 31


class BodyCategory(Enum):
    HEAD = 0
    RIGHT_ARM = 1
    RIGHT_HAND = 7
    LEFT_ARM = 2
    LEFT_HAND = 6
    TORSO = 3
    RIGHT_LEG = 4
    LEFT_LEG = 5


class Object:
    def __init__(self, id, threeD):
        self.threeD = threeD
        self.id = id


# BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139, 34, 104)]
dotColors = [(0, 0, 139), (20, 128, 48), (71, 130, 170), (205, 95, 58), (205, 150, 205)]

################################################################################

# drawing utils


def convert2D(point3D, cameraMatrix, dist):
    point, _ = cv2.projectPoints(
        np.array(point3D),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        cameraMatrix,
        dist,
    )

    return point[0][0]


################################################################################

# random utils


def convertTimestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def get_frame_bin(frame):
    return frame // 30
