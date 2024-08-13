from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    Vectors3DInterface,
)
from mmdemo.utils.cone_shape import ConeShape
from mmdemo.utils.support_utils import Joint
from mmdemo.utils.threeD_object_loc import checkBlocks
from mmdemo.utils.twoD_object_loc import convert2D


# this is for gaze body tracking, rgb gaze will be different
@final
class Gaze(BaseFeature):
    """
    Feature so get and track the gaze of participants.

    The input feature is `BaseFeature' which is the base class all features in the demo must implement.

    The output inteface is `Vectors3D`.
    """

    @classmethod
    def get_input_interfaces(cls):
        """
        Returns a list of input interface dependencies that this feature relies on.

        Arguments:
        `cls` -- instance of the Gaze feature class.

        Dependencies: `BodyTrackingInterface`, `CameraCalibration`
        """
        return [
            BodyTrackingInterface,
            CameraCalibrationInterface,
            ColorImageInterface,
            DepthImageInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        """
        Returns the output interface used to create the gaze vectors for participants.

        Arguments:
        `cls` -- instance of the Gaze feature class.

        Output:
        'Vector3DInterface' -- A collection of points for drawing the gaze's vector
        """
        return Vectors3DInterface

    def initialize(self):
        """
        Initializes the
        """
        input_interfaces = self.get_input_interfaces()
        self.body_tracking = input_interfaces[0]
        self.camera_calibration = input_interfaces[1]
        self.color = input_interfaces[2]
        self.depth = input_interfaces[3]

    def get_output(
        self,
        bt: BodyTrackingInterface,
        cc: CameraCalibrationInterface,
        col: ColorImageInterface,
        dep: DepthImageInterface,
    ) -> Vectors3DInterface | None:
        """
        Returns the output interface used to create the gaze vectors for participants.

        Arguments:
        `cls` -- instance of the Gaze feature class.

        Output:
        'Vector3DInterface' -- A collection of points for drawing the gaze's vector
        """
        if not bt.is_new() and not cc.is_new and not col.is_new() and not dep.is_new():
            return None
        for body in self.body_tracking.bodies:
            nose = self.get_joint(Joint.NOSE, body)

            ear_left = self.get_joint(Joint.EAR_LEFT, body)
            ear_right = self.get_joint(Joint.EAR_RIGHT, body)
            ear_center = (ear_left + ear_right) / 2

            eye_left = self.get_joint(Joint.EYE_LEFT, body)
            eye_right = self.get_joint(Joint.EYE_RIGHT, body)

            dir = nose - ear_center
            dir /= np.linalg.norm(nose - ear_center)

            origin = (eye_left + eye_right + nose) / 3

            p1_3d = origin
            p2_3d = origin + 1000 * dir

            cone = ConeShape(
                p1_3d,
                p2_3d,
                80,
                100,
                self.camera_calibration.cameraMatrix,
                self.camera_calibration.dist,
            )
            cone.projectRadiusLines(self.shift, self.color.frame, False, False, True)

            p1 = convert2D(
                p1_3d,
                self.camera_calibration.cameraMatrix,
                self.camera_calibration.dist,
            )
            p2 = convert2D(
                p2_3d,
                self.camera_calibration.cameraMatrix,
                self.camera_calibration.dist,
            )
            cv.line(
                self.color.frame, p1.astype(int), p2.astype(int), (255, 107, 170), 2
            )

            # TODO: Check how blocks and blockStatus will be handled.
            # targets = checkBlocks(blocks, blockStatus, self.camera_calibration.cameraMatrix, self.camera_calibration.dist, self.depth.frame, cone, self.color.frame, None, True)

            # descriptions = []
            # for t in targets:
            #     descriptions.append(t.description)
        return Vectors3DInterface(vectors=[p1, p2])

    def world_to_camera_coords(self, r_w):
        return (
            np.dot(self.camera_calibration.rotation, r_w)
            + self.camera_calibration.translation
        )

    def get_joint(self, joint, body):
        r_w = np.array(body["joint_positions"][joint.value])
        return self.world_to_camera_coords(r_w)

    # def processFrame(self, blocks, blockStatus):
    #     for body in self.body_tracking.bodies:
    #         nose = self.get_joint(Joint.NOSE, body, self.camera_calibration.rotation, self.camera_calibration.translation)

    #         ear_left = self.get_joint(Joint.EAR_LEFT, body, self.camera_calibration.rotation, self.camera_calibration.translation)
    #         ear_right = self.get_joint(Joint.EAR_RIGHT, body, self.camera_calibration.rotation, self.camera_calibration.translation)
    #         ear_center = (ear_left + ear_right) / 2

    #         eye_left = self.get_joint(Joint.EYE_LEFT, body, self.camera_calibration.rotation, self.camera_calibration.translation)
    #         eye_right = self.get_joint(Joint.EYE_RIGHT, body, self.camera_calibration.rotation, self.camera_calibration.translation)

    #         dir = nose - ear_center
    #         dir /= np.linalg.norm(nose - ear_center)

    #         origin = (eye_left + eye_right + nose) / 3

    #         p1_3d = origin
    #         p2_3d = origin + 1000*dir

    #         cone = ConeShape(p1_3d, p2_3d, 80, 100, self.camera_calibration.cameraMatrix, self.camera_calibration.dist)
    #         cone.projectRadiusLines(self.shift, self.color.frame, False, False, True)

    #         p1 = convert2D(p1_3d, self.camera_calibration.cameraMatrix, self.camera_calibration.dist)
    #         p2 = convert2D(p2_3d, self.camera_calibration.cameraMatrix, self.camera_calibration.dist)
    #         cv.line(self.color.frame, p1.astype(int), p2.astype(int), (255, 107, 170), 2)

    #         targets = checkBlocks(blocks, blockStatus, self.camera_calibration.cameraMatrix, self.camera_calibration.dist, self.depth.frame, cone, self.color.frame, None, True)

    #         descriptions = []
    #         for t in targets:
    #             descriptions.append(t.description)
