from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import BodyTrackingInterface, PoseInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Pose(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [BodyTrackingInterface]

    @classmethod
    def get_output_interface(cls):
        return PoseInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: BodyTrackingInterface):
        if not t.is_new():
            return None

        # call _, create interface, and return
