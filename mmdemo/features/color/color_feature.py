from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Color(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return []

    @classmethod
    def get_output_interface(cls):
        return ColorImageInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: None):
        if not t.is_new():
            return None

        # call __, create interface, and return
