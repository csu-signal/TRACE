import math

import cv2
import numpy as np
from numpy.linalg import norm

from mmdemo.interfaces import CameraCalibrationInterface, DepthImageInterface
from mmdemo.utils.coordinates import pixel_to_camera_3d
from mmdemo.utils.point_vector_logic import getDirectionalVector, getVectorPoint
from mmdemo.utils.support_utils import ParseResult


def distance3D(point1, point2):
    # TODO fix depth with the z channel
    # print(point1)
    # print(point2)
    # return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2) + ((point2[2] - point1[2]) ** 2))
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))


# get point on vector perpendicular to block
def projectedPoint(p1, p2, p3):
    P12 = getDirectionalVector(p2, p1)  # pointing vector
    P13 = getDirectionalVector(p3, p1)  # vertex to point

    proj_13over12 = np.dot(P13, P12) * P12 / norm(P12) ** 2
    perpendicular = proj_13over12 - P13

    return getVectorPoint(p3, perpendicular)


class TargetDescription:
    def __init__(self, description, distance):
        self.description = description
        self.distance = distance


def sortTarget(target):
    return target.distance


def checkBlocks(
    blocks,
    blockStatus,
    calibration: CameraCalibrationInterface,
    depth: DepthImageInterface,
    cone,
    frame,
    shift,
    gaze,
    gesture=False,
    index=(0, 0, 0),
):
    targets = []
    for block in blocks:
        targetPoint = [
            (block.p1[0] + block.p2[0]) // 2,
            (block.p1[1] + block.p2[1]) // 2,
        ]
        # print("Check Block: " + str(block.description))

        try:
            object3D = pixel_to_camera_3d(targetPoint, depth, calibration)
        except:
            continue

        if block.description not in blockStatus:
            cv2.rectangle(
                frame,
                (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                color=(255, 255, 255),
                thickness=3,
                shift=shift,
            )

        block.target, distance = cone.ContainsPoint(
            object3D[0],
            object3D[1],
            object3D[2],
            frame,
            False,
            gesture=gesture,
            index=index,
        )
        if block.target:
            width = 5
            targets.append(TargetDescription(block.description, distance))
            if gaze:
                if block.description not in blockStatus:
                    blockStatus[block.description] = 1
                else:
                    blockStatus[block.description] += 1
                width *= blockStatus[block.description]

            cv2.rectangle(
                frame,
                (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                color=(0, 255, 0),
                thickness=width,
                shift=shift,
            )
            cv2.circle(
                frame,
                (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)),
                radius=10,
                color=(0, 0, 0),
                thickness=10,
                shift=shift,
            )
        else:
            if block.description not in blockStatus:
                cv2.rectangle(
                    frame,
                    (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                    (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                    color=(0, 0, 255),
                    thickness=5,
                    shift=shift,
                )
                cv2.circle(
                    frame,
                    (
                        int(targetPoint[0] * 2**shift),
                        int(targetPoint[1] * 2**shift),
                    ),
                    radius=10,
                    color=(0, 0, 0),
                    thickness=10,
                    shift=shift,
                )
    targets = sorted(targets, key=sortTarget)
    return targets
