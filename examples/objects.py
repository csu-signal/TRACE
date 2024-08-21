import cv2 as cv
import numpy as np
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

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

        # output_frame = cv.resize(output_frame, (output_frame.shape[1] // 4, output_frame.shape[0] // 4))

        cv.imshow("", output_frame)
        cv.waitKey(1)


if __name__ == "__main__":
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=r"C:\Users\brady\Desktop\Group_01-master.mkv",
        playback_frame_rate=5,
    )
    objects = Object(color, depth, calibration)
    output = ShowOutput(color, objects)

    Demo(targets=[output]).run()
