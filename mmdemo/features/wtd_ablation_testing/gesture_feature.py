from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, SelectedObjectsInterface


@final
class GestureAblation(BaseFeature[SelectedObjectsInterface]):
    """
    TODO: docstring
    """

    def __init__(
        self, color: BaseFeature[ColorImageInterface], *, csv_path: Path
    ) -> None:
        super().__init__(color)
        self.csv_path = csv_path

    def initialize(self):
        # TODO: read input csv and load data
        pass

    def get_output(self, color: ColorImageInterface) -> SelectedObjectsInterface | None:
        pass
