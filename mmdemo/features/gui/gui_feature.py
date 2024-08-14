from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (  # ASRInterface,; BodyTrackingInterface,; CommonGroundInterface,; DenseParaphraseInterface,; DepthImageInterface,; GazeInterface,; GestureInterface,; MoveInterface,; ObjectInterface,; OutputFrameInterface,; PropositionInterface,; SelectedObjectsInterface,; UtteranceChunkInterface,
    ColorImageInterface,
    EmptyInterface,
    TranscriptionInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class GUI(BaseFeature[EmptyInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return [ColorImageInterface]

    @classmethod
    def get_output_interface(cls):
        return EmptyInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):  # None or .. ?
        if not t.is_new():
            return None

        # call __, create interface, and return
