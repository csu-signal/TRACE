from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features import DisplayFrame
from mmdemo.interfaces import BodyTrackingInterface, EmptyInterface


class PrintNumBodies(BaseFeature[EmptyInterface]):
    """
    Input interface is `BodyTrackingInterface`
    """

    def get_output(self, bt: BodyTrackingInterface):
        print(len(bt.bodies), "bodies detected")


if __name__ == "__main__":
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.CAMERA, camera_index=0
    )

    demo = Demo(targets=[DisplayFrame(color), PrintNumBodies(body_tracking)])
    demo.show_dependency_graph()
    demo.run()
