from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import EmptyInterface, OutputFrameInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class GUI(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [OutputFrameInterface]

    @classmethod
    def get_output_interface(cls):
        return EmptyInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: OutputFrameInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return
