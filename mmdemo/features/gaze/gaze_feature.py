from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import BodyTrackingInterface, GazeInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Gaze(BaseFeature):
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies([BodyTrackingInterface], args)

    @classmethod
    def get_output_interface(cls):
        return GazeInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: BodyTrackingInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return
