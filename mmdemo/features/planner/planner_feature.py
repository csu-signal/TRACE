from pathlib import Path
from typing import final
import re
import time
import threading

from mmdemo.features.planner.planner import create_planner, update_block_weight, check_solution

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import PlannerInterface, CommonGroundInterface

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
    
    def initialize(self):

        # Call the function to execute the planning
        start = time.time()
        self.problem, self.planner, self.actual_weight, self.believed_weight, self.blocks, self.participants, self.weights = create_planner()
        end = time.time()
        print("initializing planner took ", end-start)

    def run_check_solution(self):
        """Runs check_solution in a separate thread and stores the result."""
        solv, plan = check_solution(self.problem, self.planner)
        with self.lock:
            self.solution_result = (solv, plan)


    def get_output(
            self,
            cg: CommonGroundInterface
    ):
        
        if not cg.is_new():
            return None
        start = time.time()
        # possibly need to reinitialize the planner every time in case a prop is removed
        ebank, fbank = cg.ebank, cg.fbank
        for prop in ebank.union(fbank):
            prop_list = [p.strip() for p in prop.split(',')]
            for p in prop_list:
                block, weight = re.split(r'(?<!=)=(?!=)|<|>|!=', p)
                block, weight = block.strip(), weight.strip()
                update_block_weight(self, block, weight)

        check_thread = threading.Thread(target=self.run_check_solution)
        check_thread.start()
        check_thread.join()

        with self.lock:
            if self.solution_result:
                solv, plan = self.solution_result
                end = time.time()
                print("getting output from planner took ", end - start)
                self.solution_result = None  # Reset for next call
                return PlannerInterface(solv, plan)

        return None