import csv
from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    DepthImageInterface,
    ObjectInterface3D,
)
from mmdemo.interfaces.data import GamrTarget, ObjectInfo2D, ObjectInfo3D
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d


@final
class ObjectGroundTruth(BaseFeature[ObjectInterface3D]):
    """
    Ground truth for objects are in the frame. Given an input frame count,
    this feature will return the most recent objects that have not already
    been returned.

    The input interfaces are `DepthImageInterface` and `CameraCalibrationInterface`.

    The output interface is `ObjectInterface3D`. The 3d center coordinates will
    only be reliable if the input depth and calibration are from the same
    recording as the annotations.

    Keyword arguments:
    `csv_path` -- path to the WTD annotation objects.csv file
    """

    def __init__(
        self,
        depth: BaseFeature[DepthImageInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        *,
        csv_path: Path,
    ):
        super().__init__(depth, calibration)
        self.csv_path = csv_path

    def initialize(self):
        self.data = ObjectGroundTruth.read_csv(self.csv_path)
        self.current_frame = 0

    def get_output(
        self, depth: DepthImageInterface, calibration: CameraCalibrationInterface
    ) -> ObjectInterface3D | None:
        if not depth.is_new():
            return None

        # no data currently found
        last_frame_with_data = None

        # loop from the last frame count received to the current one
        while self.current_frame <= depth.frame_count:
            # update last frame with data if it exists
            if self.current_frame in self.data:
                last_frame_with_data = self.current_frame

            self.current_frame += 1

        # if we didn't find any data, return no objects
        if last_frame_with_data is None:
            return ObjectInterface3D(objects=[])

        objects: list[ObjectInfo3D] = []
        for o in self.data[last_frame_with_data]:
            # calculate 3d center using calibration and depth info
            center_2d = [(o.p1[0] + o.p2[0]) // 2, (o.p1[1] + o.p2[1]) // 2]
            try:
                center = pixel_to_camera_3d(center_2d, depth, calibration)
            except CoordinateConversionError:
                continue

            objects.append(
                ObjectInfo3D(
                    p1=o.p1, p2=o.p2, center=tuple(center), object_class=o.object_class
                )
            )

        return ObjectInterface3D(objects=objects)

    @staticmethod
    def read_csv(path):
        """
        Returns a dict[int, list[ObjectInfo2D]] mapping frame counts to
        lists of object info.
        """
        data_by_frame: dict[int, list[ObjectInfo2D]] = {}

        with open(path, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i: j for i, j in zip(keys, row)}

                frame_index = int(data["frame_index"])
                if frame_index not in data_by_frame:
                    data_by_frame[frame_index] = []

                p1 = (int(data["p10"]), int(data["p11"]))
                p2 = (int(data["p20"]), int(data["p21"]))
                object_class = ObjectGroundTruth.cls_to_target(int(data["class"]))
                data_by_frame[frame_index].append(
                    ObjectInfo2D(p1=p1, p2=p2, object_class=object_class)
                )

        return data_by_frame

    @staticmethod
    def cls_to_target(c: int) -> GamrTarget:
        match c:
            case 0:
                return GamrTarget.RED_BLOCK
            case 1:
                return GamrTarget.YELLOW_BLOCK
            case 2:
                return GamrTarget.GREEN_BLOCK
            case 3:
                return GamrTarget.BLUE_BLOCK
            case 4:
                return GamrTarget.PURPLE_BLOCK
            case 5:
                return GamrTarget.SCALE
            case _:
                return GamrTarget.UNKNOWN
