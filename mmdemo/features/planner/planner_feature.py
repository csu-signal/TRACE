from pathlib import Path
from typing import final
import re
import time
import threading

from mmdemo.features.planner.planner import create_planner, update_block_weight, check_solution

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import PlannerInterface, CommonGroundInterface, EmptyInterface

@final
class Planner(BaseFeature[PlannerInterface]):
    """
    Detemine if the task is still solveable.
    
    Input interfaces is `CommonGroundInterface`.
    Output interface is `PlannerInterface`.
    """

    def __init__(
            self,
            common_ground: BaseFeature[CommonGroundInterface],
            planner_path: Path | None = None
    ) -> None:
        
        super().__init__(common_ground)
        self.planner_path = planner_path
        self.lock = threading.Lock()
        self.solution_result = True, ""
    
    def initialize(self):

        # start = time.time()
        self.problem, self.planner, self.actual_weight, self.believed_weight, self.blocks, self.participants, self.weights = create_planner()
        # end = time.time()
        # print("initializing planner took ", end-start)

    def run_check_solution(self):
        """Runs check_solution in a separate thread and stores the result."""
        solv, plan = check_solution()
        with self.lock:
            self.solution_result = (solv, plan)


    def get_output(
            self,
            cg: CommonGroundInterface
    ):
        
        if cg.is_new() and not cg == EmptyInterface():
            start = time.time()
            # ebank, fbank = cg.ebank, cg.fbank
            self.fbank = cg.fbank
            for prop in self.fbank:
                prop_list = [p.strip() for p in prop.split(',')]
                for p in prop_list:
                    block, weight = re.split(r'(?<!=)=(?!=)|<|>|!=', p)
                    block, weight = block.strip(), weight.strip()
                    update_block_weight(self, block, weight)

            check_thread = threading.Thread(target=self.run_check_solution)
            check_thread.start()
            end = time.time()
            # print("planner output took ", end-start)
            with self.lock:
                if self.solution_result is not None:
                    solv, plan = self.solution_result
                    print("Plan is: ", plan) #getting empty plan 
                    return PlannerInterface(solv, plan, cg.fbank)

        return None