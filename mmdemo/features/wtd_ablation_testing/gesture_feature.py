import csv
import json
from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, SelectedObjectsInterface
from mmdemo.interfaces.data import GamrTarget, ObjectInfo2D


@final
class GestureSelectedObjectsGroundTruth(BaseFeature[SelectedObjectsInterface]):
    """
    Ground truth for which objects are being selected by gesture. Given
    an input frame count, this feature will return the most recent
    gesture selection that has not already been returned.

    The input interface is `ColorImageInterface`.

    The output interface is `SelectedObjectsInterface`. The selected object
    classes will be correct, but the object locations will not. Thus, do not
    use this feature as input to another feature which requires correct
    object locations.

    Keyword arguments:
    `csv_path` -- path to the WTD annotation gestures.csv file
    """

    def __init__(
        self, color: BaseFeature[ColorImageInterface], *, csv_path: Path
    ) -> None:
        super().__init__(color)
        self.csv_path = csv_path

    def initialize(self):
        self.data = GestureSelectedObjectsGroundTruth.read_csv_as_dict(self.csv_path)
        self.current_frame = 0

    def get_output(self, color: ColorImageInterface) -> SelectedObjectsInterface | None:
        if not color.is_new():
            return None

        # no data currently found
        last_frame_with_data = None

        # loop from the last frame count received to the current one
        while self.current_frame <= color.frame_count:
            # update last frame with data if it exists
            if self.current_frame in self.data:
                last_frame_with_data = self.current_frame

            self.current_frame += 1

        # if we didn't find any data, return no objects
        if last_frame_with_data is None:
            return SelectedObjectsInterface(objects=[])

        objects: list[tuple[ObjectInfo2D, bool]] = []
        for i in self.data[last_frame_with_data]:
            objects.append(
                (
                    ObjectInfo2D(p1=(0, 0), p2=(0, 0), object_class=i),
                    True,
                )
            )

        return SelectedObjectsInterface(objects=objects)

    @staticmethod
    def read_csv_as_dict(path):
        """
        Returns a dict[int, list[GamrTarget]] mapping frame counts to
        selected object targets.
        """
        data_by_frame: dict[int, list[GamrTarget]] = {}

        with open(path, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i: j for i, j in zip(keys, row)}
                targets: list[GamrTarget] = []
                for t in json.loads(data["blocks"]):
                    targets.append(GestureSelectedObjectsGroundTruth.str_to_target(t))
                data_by_frame[int(data["frame"])] = targets

        return data_by_frame

    @staticmethod
    def str_to_target(s: str) -> GamrTarget:
        match s:
            case "red":
                return GamrTarget.RED_BLOCK
            case "blue":
                return GamrTarget.BLUE_BLOCK
            case "green":
                return GamrTarget.GREEN_BLOCK
            case "purple":
                return GamrTarget.PURPLE_BLOCK
            case "yellow":
                return GamrTarget.YELLOW_BLOCK
            case _:
                return GamrTarget.UNKNOWN
