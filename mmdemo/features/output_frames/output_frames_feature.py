from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import (  # FrameCountInterface,; GazeInterface,
    ColorImageInterface,
    CommonGroundInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    Vectors3DInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class OutputFrames(BaseFeature[ColorImageInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return [
            ColorImageInterface,
            # FrameCountInterface,
            CommonGroundInterface,
            Vectors3DInterface,
            SelectedObjectsInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return ColorImageInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(
        self,
        *args: BaseInterface,
        # gaze: GazeInterface,
        tran: TranscriptionInterface,
    ):
        for feature in args:
            # if isinstance(feature, FrameCountInterface):
            # draw frame count
            pass

        # if not gaze.is_new() and not tran.is_new():
        #     return None

        # call _, create interface, and return
