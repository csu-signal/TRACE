from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ASRInterface,
    BodyTrackingInterface,
    ColorImageInterface,
    CommonGroundInterface,
    DenseParaphraseInterface,
    DepthImageInterface,
    EmptyInterface,
    GazeInterface,
    GestureInterface,
    MoveInterface,
    ObjectInterface,
    OutputFrameInterface,
    PropositionInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    UtteranceChunkInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class GUI(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [
            OutputFrameInterface,
            CommonGroundInterface,
            MoveInterface,
            PropositionInterface,
            DenseParaphraseInterface,
            SelectedObjectsInterface,
            GazeInterface,
            ObjectInterface,
            GestureInterface,
            ColorImageInterface,
            DepthImageInterface,
            BodyTrackingInterface,
            ASRInterface,
            UtteranceChunkInterface,
            TranscriptionInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return EmptyInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: None):  # None or .. ?
        if not t.is_new():
            return None

        # call __, create interface, and return
