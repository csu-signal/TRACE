from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import TranscriptionInterface, _

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Evaluation(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return []

    @classmethod
    def get_output_interface(cls):
        return _

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return

    # class EvaluationConfig:
    # """
    # directory: where to load output csvs from
    # objects: if the object feature should be loaded from csv
    # gesture: if the gesture feature should be loaded from csv
    # asr: if the asr feature should be loaded from csv
    # prop: if the prop feature should be loaded from csv
    # move: if the move feature should be loaded from csv
    # prop_model: the path to the proposition extrator model to use
    # move_model: the path to the move classifier model to use
    # fallback_audio: if asr is loaded from a csv, use this file to add audio to final.mp4
    # """
    # directory: str | Path
    # objects: bool = False
    # gesture: bool = False
    # asr: bool = False
    # prop: bool = False
    # move: bool = False
    # prop_model: str | None = None
    # move_model: str | None = None
    # fallback_audio: str | Path | None = None  # add to processed frames in the end
