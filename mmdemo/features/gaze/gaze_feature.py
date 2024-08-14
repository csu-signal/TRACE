from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    GazeConesInterface,
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.support_utils import Joint
from mmdemo.utils.twoD_object_loc import convert2D


# this is for gaze body tracking, rgb gaze will be different
@final
class Gaze(BaseFeature[GazeConesInterface]):
    """
    A feature to get and track the points of participants' gaze vectors.

    Input feature is `BaseFeature' which is the base class all features in the demo must implement.

    Output inteface is `GazeConesInterface`.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_input_interfaces(cls):
        return [
            BodyTrackingInterface,
            CameraCalibrationInterface,
            ColorImageInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return GazeConesInterface

    def initialize(self):
        pass

    def get_output(
        self,
        bt: BodyTrackingInterface,
        cc: CameraCalibrationInterface,
        col: ColorImageInterface,
    ) -> GazeConesInterface | None:
        if not bt.is_new() and not cc.is_new() and not col.is_new():
            return None
        cones = []
        body_ids = []
        bod_id = 1
        for body in bt.bodies:
            nose = self.get_joint(Joint.NOSE, body, cc)

            ear_left = self.get_joint(Joint.EAR_LEFT, body, cc)
            ear_right = self.get_joint(Joint.EAR_RIGHT, body, cc)
            ear_center = (ear_left + ear_right) / 2

            eye_left = self.get_joint(Joint.EYE_LEFT, body, cc)
            eye_right = self.get_joint(Joint.EYE_RIGHT, body, cc)

            dir = nose - ear_center
            dir /= np.linalg.norm(nose - ear_center)

            origin = (eye_left + eye_right + nose) / 3

            origin_point = origin
            end_point = origin + 1000 * dir

            cone = Cone(origin_point, end_point, 80, 100)
            cones.append(cone)
            body_ids.append(bod_id)
            bod_id += 1
            # p1 = convert2D(
            #     p1_3d,
            #     cc.cameraMatrix,
            #     cc.distortion,
            # )
            # p2 = convert2D(
            #     p2_3d,
            #     cc.cameraMatrix,
            #     cc.distortion,
            # )

        return GazeConesInterface(body_ids=body_ids, cones=cones)

    @final
    def get_joint(self, joint, body, cc):
        """
        `self` -- instance of Gaze class
        `joint` -- the joint to be retrieved
        `body` -- the body whose joint is being retrieved
        `cc` -- instance of `CameraCalibrationInterface`

        Returns camera coordinates of requested joint
        """
        r_w = np.array(body["joint_positions"][joint.value])
        return np.dot(cc.rotation, r_w) + cc.translation
