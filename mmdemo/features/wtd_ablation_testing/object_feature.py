from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import DepthImageInterface, ObjectInterface3D


@final
class ObjectAblation(BaseFeature[ObjectInterface3D]):
    """
    TODO: docstring
    """

    def __init__(
        self, depth: BaseFeature[DepthImageInterface], *, csv_path: Path
    ) -> None:
        super().__init__(depth)
        self.csv_path = csv_path

    def initialize(self):
        # TODO: read input csv and load data
        pass

    def get_output(self, depth: DepthImageInterface) -> ObjectInterface3D | None:
        pass
