import cv2 as cv
import numpy as np
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    EmptyInterface,
    GazeConesInterface,
    GestureConesInterface,
    ObjectInterface3D,
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import camera_3d_to_pixel


class ShowOutput(BaseFeature[EmptyInterface]):
    def get_output(
        self,
        color: ColorImageInterface,
        objects: ObjectInterface3D,
        gaze: GazeConesInterface,
        gesture: GestureConesInterface,
        calibration: CameraCalibrationInterface,
    ):
        if not color.is_new() or not objects.is_new():
            return None

        output_frame = np.copy(color.frame)
        output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

        for block in objects.objects:
            c = (0, 0, 255)
            cv.rectangle(
                output_frame,
                (int(block.p1[0]), int(block.p1[1])),
                (int(block.p2[0]), int(block.p2[1])),
                color=c,
                thickness=5,
            )

        for cone in gaze.cones + gesture.cones:
            self.draw_cone(output_frame, cone, calibration)

        cv.imshow("", output_frame)
        cv.waitKey(1)

    def draw_cone(self, im, cone: Cone, calibartion: CameraCalibrationInterface):
        p1 = camera_3d_to_pixel(cone.base, calibartion)
        p2 = camera_3d_to_pixel(cone.vertex, calibartion)
        cv.line(im, p1, p2, (255, 0, 0), thickness=3)


if __name__ == "__main__":
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA,
        camera_index=0
    )
    objects = Object(color, depth, calibration)
    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)
    output = ShowOutput(color, objects, gaze, gesture, calibration)

    Demo(targets=[output]).run()
