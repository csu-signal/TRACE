from enum import Enum


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
