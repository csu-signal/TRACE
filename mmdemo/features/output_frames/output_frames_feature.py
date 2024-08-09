from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    GazeInterface,
    OutputFrameInterface,
    TranscriptionInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class OutputFrames(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [TranscriptionInterface, GazeInterface]

    @classmethod
    def get_output_interface(cls):
        return OutputFrameInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(
        self,
        gaze: GazeInterface,
        tran: TranscriptionInterface,
    ):
        if not gaze.is_new() and not tran.is_new():
            return None

        # call _, create interface, and return
