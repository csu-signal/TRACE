from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    GazeConesInterface,
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import world_3d_to_camera_3d
from mmdemo.utils.joints import Joint


@final
class GazeBodyTracking(BaseFeature[GazeConesInterface]):
    """
    A feature to get and track the points of participants' gaze vectors using
    Azure Kinect body tracking data.

    Input interfaces are `BodyTrackingInterface` and `CameraCalibrationInterface`.

    Output interface is `GazeConesInterface`.
    """

    BASE_RADIUS = 80
    VERTEX_RADIUS = 100

    def __init__(
        self,
        bt: BaseFeature[BodyTrackingInterface],
        cal: BaseFeature[CameraCalibrationInterface],
    ) -> None:
        super().__init__(bt, cal)

    def get_output(
        self,
        bt: BodyTrackingInterface,
        cc: CameraCalibrationInterface,
    ) -> GazeConesInterface | None:
        if not bt.is_new():
            return None
        cones = []
        body_ids = []

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

            cone = Cone(origin_point, end_point, self.BASE_RADIUS, self.VERTEX_RADIUS)
            cones.append(cone)
            body_ids.append(body["body_id"])

        return GazeConesInterface(azure_body_ids=body_ids, wtd_body_ids=[], cones=cones)

    @final
    def get_joint(self, joint, body, cc):
        """
        `joint` -- the joint to be retrieved
        `body` -- the body whose joint is being retrieved
        `cc` -- instance of `CameraCalibrationInterface`

        Returns camera coordinates of requested joint
        """
        r_w = np.array(body["joint_positions"][joint.value])
        return world_3d_to_camera_3d(r_w, cc)
