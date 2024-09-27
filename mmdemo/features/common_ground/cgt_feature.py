from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.features.common_ground.closure_rules import CommonGround
from mmdemo.interfaces import CommonGroundInterface, MoveInterface, PropositionInterface


@final
class CommonGroundTracking(BaseFeature[CommonGroundInterface]):
    """
    Track the common ground of participants solving the Weights Task.

    Input interfaces are `MoveInterface` and `PropositionInterface`

    Output interface is `CommonGroundInterface`
    """

    def __init__(
        self, move: BaseFeature[MoveInterface], prop: BaseFeature[PropositionInterface]
    ):
        super().__init__(move, prop)

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
