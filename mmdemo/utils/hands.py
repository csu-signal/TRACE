import itertools
from enum import Enum

import numpy as np

from mmdemo.interfaces import CameraCalibrationInterface
from mmdemo.utils.coordinates import camera_3d_to_pixel, world_3d_to_camera_3d
from mmdemo.utils.joints import BodyCategory, Joint, getPointSubcategory


# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
class Handedness(Enum):
    Left = "Right"
    Right = "Left"


def normalize_landmarks(hand_landmarks, image_width, image_height):
    # TODO: Hannah maybe you could add to the docstring to explain what this normalization does?
    # there were a few really short functions that were only used once so I combined them here
    # feel free to separate it out again if you want.
    # I did my best effort at adding comments but they might be wrong
    """
    Normalize landmarks for the pointing detector
    """

    normalized_landmarks = []

    # scale landmarks to image size
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        normalized_landmarks.append([landmark_x, landmark_y])

    # shift landmarks to put first landmark at origin
    base_x, base_y = normalized_landmarks[0][0], normalized_landmarks[0][1]
    for index in range(len(normalized_landmarks)):
        normalized_landmarks[index][0] = normalized_landmarks[index][0] - base_x
        normalized_landmarks[index][1] = normalized_landmarks[index][1] - base_y

    # Convert to a one-dimensional list
    normalized_landmarks = list(itertools.chain.from_iterable(normalized_landmarks))

    # normalize landmarks by dividing by max absolute value
    max_value = max(list(map(abs, normalized_landmarks)))
    normalized_landmarks = list(map(lambda x: x / max_value, normalized_landmarks))

    return normalized_landmarks


def get_average_hand_pixel(
    body: dict, calibration: CameraCalibrationInterface, handedness: Handedness
):
    vals = []

    if handedness == Handedness.Left:
        category = BodyCategory.LEFT_HAND
    else:
        category = BodyCategory.RIGHT_HAND

    for jointIndex, joint in enumerate(body["joint_positions"]):
        bodyLocation = getPointSubcategory(Joint(jointIndex))
        if bodyLocation == category:
            points_camera_3D = world_3d_to_camera_3d(joint, calibration)
            points2D = camera_3d_to_pixel(points_camera_3D, calibration)
            vals.append(points2D)

    return np.mean(vals, axis=0)


def createBoundingBox(center, w, h):
    offset = np.array([w, h]) / 2
    return np.array([center - offset, center + offset], dtype=np.int64)