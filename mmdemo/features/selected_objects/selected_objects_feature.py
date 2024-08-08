from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import SelectedObjectsInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class SelectedObjects(BaseFeature):
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies([TranscriptionInterface], args)

    @classmethod
    def get_output_interface(cls):
        return SelectedObjectsInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call _, create interface, and return
