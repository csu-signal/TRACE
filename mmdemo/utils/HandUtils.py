import copy
import itertools

import cv2
import numpy as np

from mmdemo.utils.utils import BodyCategory, Joint


def processHands(image, hand):
    normalized = []
    landmarkArray = calc_landmark_list(image, hand)
    for landmark in pre_process_landmark(landmarkArray):
        normalized.append(landmark)
    return normalized


def getAverageHandLocations(body, w, h, rotation, translation, cameraMatrix, dist):
    hand_left_x_max = 0
    hand_left_y_max = 0
    hand_left_x_min = w
    hand_left_y_min = h

    hand_right_x_max = 0
    hand_right_y_max = 0
    hand_right_x_min = w
    hand_right_y_min = h

    rightXTotal = 0
    leftXTotal = 0
    rightYTotal = 0
    leftYTotal = 0

    # print(body["joint_positions"])
    for jointIndex, joint in enumerate(body["joint_positions"]):
        bodyLocation = getPointSubcategory(Joint(jointIndex))
        if bodyLocation == BodyCategory.LEFT_HAND:  # this should really be a method
            points2D, _ = cv2.projectPoints(
                np.array(joint), rotation, translation, cameraMatrix, dist
            )

            x = points2D[0][0][0]
            y = points2D[0][0][1]
            leftXTotal += x
            leftYTotal += y

            if x > hand_left_x_max:
                hand_left_x_max = x
            if x < hand_left_x_min:
                hand_left_x_min = x
            if y > hand_left_y_max:
                hand_left_y_max = y
            if y < hand_left_y_min:
                hand_left_y_min = y

        if bodyLocation == BodyCategory.RIGHT_HAND:
            points2D, _ = cv2.projectPoints(
                np.array(joint), rotation, translation, cameraMatrix, dist
            )

            x = points2D[0][0][0]
            y = points2D[0][0][1]
            rightXTotal += x
            rightYTotal += y
            if x > hand_right_x_max:
                hand_right_x_max = x
            if x < hand_right_x_min:
                hand_right_x_min = x
            if y > hand_right_y_max:
                hand_right_y_max = y
            if y < hand_right_y_min:
                hand_right_y_min = y

    leftXAverage = leftXTotal / 4
    leftYAverage = leftYTotal / 4
    rightXAverage = rightXTotal / 4
    rightYAverage = rightYTotal / 4

    return leftXAverage, leftYAverage, rightXAverage, rightYAverage


def createBoundingBox(xAverage, yAverage):
    # xMax = xAverage + (xAverage * 0.05)
    # xMin = xAverage - (xAverage * 0.05)
    # yMax = yAverage + (yAverage * 0.05)
    # yMin = yAverage - (yAverage * 0.05)

    xMax = xAverage + (32)
    xMin = xAverage - (32)
    yMax = yAverage + (32)
    yMin = yAverage - (32)
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    return [
        int(xMin - (xSpan)),
        int(yMin - (ySpan)),
        int(xMax + (xSpan)),
        int(yMax + (ySpan)),
    ]


def createHeadBoundingBox(xAverage, yAverage, xRange, yRange):
    xMax = xAverage + xRange
    xMin = xAverage - xRange
    yMax = yAverage + yRange
    yMin = yAverage - yRange
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    return [
        int(xMin - (xSpan)),
        int(yMin - (ySpan)),
        int(xMax + (xSpan)),
        int(yMax + (ySpan)),
    ]


def getPointSubcategory(joint):
    if (
        joint == Joint.PELVIS
        or joint == Joint.NECK
        or joint == Joint.SPINE_NAVEL
        or joint == Joint.SPINE_CHEST
    ):
        return BodyCategory.TORSO
    if (
        joint == Joint.CLAVICLE_LEFT
        or joint == Joint.SHOULDER_LEFT
        or joint == Joint.ELBOW_LEFT
    ):
        return BodyCategory.LEFT_ARM
    if (
        joint == Joint.WRIST_LEFT
        or joint == Joint.HAND_LEFT
        or joint == Joint.HANDTIP_LEFT
        or joint == Joint.THUMB_LEFT
    ):
        return BodyCategory.LEFT_HAND
    if (
        joint == Joint.CLAVICLE_RIGHT
        or joint == Joint.SHOULDER_RIGHT
        or joint == Joint.ELBOW_RIGHT
    ):
        return BodyCategory.RIGHT_ARM
    if (
        joint == Joint.WRIST_RIGHT
        or joint == Joint.HAND_RIGHT
        or joint == Joint.HANDTIP_RIGHT
        or joint == Joint.THUMB_RIGHT
    ):
        return BodyCategory.RIGHT_HAND
    if (
        joint == Joint.HIP_LEFT
        or joint == Joint.KNEE_LEFT
        or joint == Joint.ANKLE_LEFT
        or joint == Joint.FOOT_LEFT
    ):
        return BodyCategory.LEFT_LEG
    if (
        joint == Joint.HIP_RIGHT
        or joint == Joint.KNEE_RIGHT
        or joint == Joint.ANKLE_RIGHT
        or joint == Joint.FOOT_RIGHT
    ):
        return BodyCategory.RIGHT_LEG
    if (
        joint == Joint.HEAD
        or joint == Joint.NOSE
        or joint == Joint.EYE_LEFT
        or joint == Joint.EAR_LEFT
        or joint == Joint.EYE_RIGHT
        or joint == Joint.EAR_RIGHT
    ):
        return BodyCategory.HEAD


def calc_landmark_list(image, hand):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(hand.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
