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
    match joint:
        case Joint.PELVIS | Joint.NECK | Joint.SPINE_NAVEL | Joint.SPINE_CHEST:
            return BodyCategory.TORSO
        case Joint.CLAVICLE_LEFT | Joint.SHOULDER_LEFT | Joint.ELBOW_LEFT:
            return BodyCategory.LEFT_ARM
        case Joint.WRIST_LEFT | Joint.HAND_LEFT | Joint.HANDTIP_LEFT | Joint.THUMB_LEFT:
            return BodyCategory.LEFT_HAND
        case Joint.CLAVICLE_RIGHT | Joint.SHOULDER_RIGHT | Joint.ELBOW_RIGHT:
            return BodyCategory.RIGHT_ARM
        case (
            Joint.WRIST_RIGHT
            | Joint.HAND_RIGHT
            | Joint.HANDTIP_RIGHT
            | Joint.THUMB_RIGHT
        ):
            return BodyCategory.RIGHT_HAND
        case Joint.HIP_LEFT | Joint.KNEE_LEFT | Joint.ANKLE_LEFT | Joint.FOOT_LEFT:
            return BodyCategory.LEFT_LEG
        case Joint.HIP_RIGHT | Joint.KNEE_RIGHT | Joint.ANKLE_RIGHT | Joint.FOOT_RIGHT:
            return BodyCategory.RIGHT_LEG
        case (
            Joint.HEAD
            | Joint.NOSE
            | Joint.EYE_LEFT
            | Joint.EAR_LEFT
            | Joint.EYE_RIGHT
            | Joint.EAR_RIGHT
        ):
            return BodyCategory.HEAD
