from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import (
    ColorImageInterface,
    CommonGroundInterface,
    FrameCountInterface,
    GazeInterface,
    OutputFrameInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    Vectors3D,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class OutputFrames(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [
            ColorImageInterface,
            FrameCountInterface,
            CommonGroundInterface,
            Vectors3D,
            Vectors3D,
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
        gaze: GazeInterface,
        tran: TranscriptionInterface,
    ):
        for feature in args:
            if isinstance(feature, FrameCountInterface):
                # draw frame count
                pass

        if not gaze.is_new() and not tran.is_new():
            return None

        # call _, create interface, and return
