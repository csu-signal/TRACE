from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.features.common_ground.closure_rules import CommonGround
from mmdemo.interfaces import CommonGroundInterface, MoveInterface, PropositionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class CommonGroundTracking(BaseFeature[CommonGroundInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return [
            MoveInterface,
            PropositionInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return CommonGroundInterface

    def initialize(self):
        self.closure_rules = CommonGround()
        self.most_recent_prop = "no prop"

    def get_output(
        self,
        move: MoveInterface,
        prop: PropositionInterface,
    ):
        if not move.is_new() or not prop.is_new():
            return None

        prop_data = prop.prop
        move_data = move.move

        if prop_data != "no prop":
            self.most_recent_prop = prop_data

        self.closure_rules.update(move_data, self.most_recent_prop)

        return CommonGroundInterface(
            qbank=self.closure_rules.qbank,
            fbank=self.closure_rules.fbank,
            ebank=self.closure_rules.ebank,
        )
