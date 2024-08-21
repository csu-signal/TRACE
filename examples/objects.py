import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import ColorImageInterface, EmptyInterface, ObjectInterface3D


class ShowOutput(BaseFeature[EmptyInterface]):
    def get_output(self, color: ColorImageInterface, objects: ObjectInterface3D):
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


if __name__ == "__main__":
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK
    )
    objects = Object(color, depth, calibration)
    output = ShowOutput(color, objects)

    Demo(targets=[output]).run()
