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
        # self.name = name
        # self.color = color
        pass

    def get_output(self, t: None):
        if not t.is_new():
            return None

        # call __, create interface, and return

    # colors = [
    #         Color("red", (0, 0, 255)),
    #         Color("blue", (255, 0, 0)),
    #         Color("green", (19, 129, 51)),
    #         Color("purple", (128, 0, 128)),
    #         Color("yellow", (0, 215, 255))]

    # fontScales = [1.5, 1.5, 0.75, 0.5, 0.5]
    # fontThickness = [3, 3, 2, 2, 2]
