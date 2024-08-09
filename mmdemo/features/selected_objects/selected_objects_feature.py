from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ASRInterface,
    BodyTrackingInterface,
    ColorImageInterface,
    DepthImageInterface,
    GestureInterface,
    ObjectInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    UtteranceChunkInterface,
    Vectors3D,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class SelectedObjects(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [ObjectInterface, Vectors3D, Vectors3D]

    @classmethod
    def get_output_interface(cls):
        return SelectedObjectsInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(
        self,
        obj: ObjectInterface,
        gest: GestureInterface,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        bod: BodyTrackingInterface,
        asr: ASRInterface,
        utt: UtteranceChunkInterface,
        tran: TranscriptionInterface,
    ):
        if (
            not obj.is_new()
            and not gest.is_new()
            and not col.is_new()
            and not dep.is_new
            and not bod.is_new()
            and not asr.is_new()
            and not utt.is_new()
            and not tran.is_new()
        ):
            return None

        # call _, create interface, and return
