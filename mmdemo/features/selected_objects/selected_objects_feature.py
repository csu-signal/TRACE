from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (  # ASRInterface,; GestureInterface,; ObjectInterface,; UtteranceChunkInterface,
    BodyTrackingInterface,
    ColorImageInterface,
    DepthImageInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    Vectors3DInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class SelectedObjects(BaseFeature[SelectedObjectsInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return []  # [ObjectInterface, Vectors3D, Vectors3D]

    @classmethod
    def get_output_interface(cls):
        return SelectedObjectsInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(
        self,
        # obj: ObjectInterface,
        # gest: GestureInterface,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        bod: BodyTrackingInterface,
        # asr: ASRInterface,
        # utt: UtteranceChunkInterface,
        tran: TranscriptionInterface,
    ):
        if not col.is_new() or not dep.is_new or not bod.is_new() or not tran.is_new():
            return None

        # call _, create interface, and return
