from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    DenseParaphraseInterface,
    PropositionInterface,
    TranscriptionInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Proposition(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [TranscriptionInterface, DenseParaphraseInterface]

    @classmethod
    def get_output_interface(cls):
        return PropositionInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface, s: DenseParaphraseInterface):
        if not t.is_new() and not s.is_new():
            return None

        # call prop extractor, create interface, and return
