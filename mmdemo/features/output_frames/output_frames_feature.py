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
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies([TranscriptionInterface, GazeInterface], args)

    @classmethod
    def get_output_interface(cls):
        return OutputFrameInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface, s: GazeInterface):
        if not t.is_new():
            return None

        # call _, create interface, and return
