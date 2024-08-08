from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ObjectInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Object(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return []

    @classmethod
    def get_output_interface(cls):
        return ObjectInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call move classifier, create interface, and return
