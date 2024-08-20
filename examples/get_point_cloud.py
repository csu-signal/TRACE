import pickle

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

import mmdemo.features as fs
from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.interfaces import DepthImageInterface


class PrintFeature(BaseFeature):
    def initialize(self):
        self.c = 0

    def get_output(self, *args):
        if not all(i.is_new() for i in args):
            return None

        for i in args:
            if isinstance(i, DepthImageInterface):
                print("depth shape", i.frame.shape)

        if self.c in [10]:
            with open(f"frame{self.c:05}.pkl", "wb") as f:
                pickle.dump(args, f)

        self.c += 1

    def is_done(self) -> bool:
        return self.c > 10


if __name__ == "__main__":
    for i, j in zip(
        ["frame00005.pkl", "frame00010.pkl"],
        ["tests\\data\\point_cloud_01.pkl", "tests\\data\\point_cloud_02.pkl"],
    ):
        with open(f"normal_frames\\{i}", "rb") as f:
            _, depth, _, calibration = pickle.load(f)

        with open(f"point_frames\\{i}", "rb") as f:
            _, point_cloud, _, calibration = pickle.load(f)
            point_cloud = point_cloud.frame.astype(float)

        with open(j, "wb") as f:
            pickle.dump((point_cloud, depth, calibration), f)

    # color, depth, bt, calibration = create_azure_kinect_features(
    #     DeviceType.PLAYBACK,
    #     mkv_path=r"C:\Users\brady\Desktop\Group_01-sub1.mkv",
    #     mkv_frame_rate=30,
    #     playback_frame_rate=30,
    # )
    #
    # Demo(targets=[PrintFeature(color, depth, bt, calibration)]).run()
