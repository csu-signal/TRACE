from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Proposition(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return []

    @classmethod
    def get_output_interface(cls):
        return PropositionInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call prop extractor, create interface, and return
