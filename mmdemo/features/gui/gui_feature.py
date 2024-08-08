from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import EmptyInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class GUI(BaseFeature):
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies([TranscriptionInterface], args)

    @classmethod
    def get_output_interface(cls):
        return EmptyInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: EmptyInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return
