from pathlib import Path
from typing import final
import re
import time
import threading
from collections import defaultdict

from mmdemo.features.planner.lego_planner import *


from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import DpipPlannerInterface, DpipActionInterface, EmptyInterface

@final
class DpipPlanner(BaseFeature[DpipPlannerInterface]):
    """
    Detemine if the task is still solveable.
    
    Input interfaces is `DpipActionInterface`.
    Output interface is `PlannerInterface`.
    """

    def __init__(
            self,
            actions: BaseFeature[DpipActionInterface],
    ) -> None:
        super().__init__(actions = actions)
        
        self.lock = threading.Lock()
        self.solution_result = True, ""
    
    def initialize(self):
        self.block_map = self.generate_block_dict()

    def generate_block_dict():
        colors = {
            'b': "Blue",
            'g': "Green",
            'o': "Orange",
            'r': "Red",
            'y': "Yellow"
        }

        shapes = {
            'r': BlockType.RECTANGLE,
            's': BlockType.SQUARE
        }

        block_map = {}
        for color_code, color_name in colors.items():
            for shape_code, shape_enum in shapes.items():
                key = color_code + shape_code
                block_map[key] = {
                    "color": color_name,
                    "shape": shape_enum.value
                }


    def check_solution(self, json):
        """check if the current state is solvable"""
        planner = LegoPlanner(grid_width=6, grid_depth=6, grid_height=3)
        block_id = 1
        for director_key in ["D1", "D2", "D3"]:
            director_view = json[director_key]
            for row_name in ["row_0", "row_1", "row_2"]:
                row = director_view[row_name]
                for col_idx, element in enumerate(row):
                    color = element['color']
                    size = element['size']
                    if color == "unknown":
                        continue
                    # Create a unique block ID
                    unique_id = f"{director_key}_{row_name}_{col_idx}_{block_id}"
                    # Add block to planner as a square with the actual color
                    planner.add_block(Block(
                        id=unique_id,
                        color=Color[color.capitalize()],
                        block_type=BlockType.SQUARE # For speed; use block_map[block]["shape"] for fidelity
                    ))
                    block_id += 1
                

        data_entries = []
        for item in description.split("; "):
            if item:
                parts = item.split(" ")
                block = parts[0]
                pos = parts[2].strip("()")
                data_entries.append((block, pos))

        xy_dict = defaultdict(list)
        for block, pos in data_entries:
            x, y, z = map(int, pos.split(","))
            xy_dict[(2*x, 2*y)].append((block, z))

        normalized_entries = []
        for (x, y), blocks in xy_dict.items():
            blocks_sorted = sorted(blocks, key=lambda b: b[1], reverse=True)[:3]
            for norm_z, (block, _) in enumerate(reversed(blocks_sorted)):
                normalized_entries.append([block, x, y, norm_z])
                
        new_blocks = [
            {
                "id": i,
                "color": self.block_map[block]["color"],
                "shape": BlockType.SQUARE.value  # For speed; use block_map[block]["shape"] for fidelity
            }
            for i, (block, x, y, norm_z) in enumerate(normalized_entries, 1)
        ]

        new_constraints = [
            f"{i} at {x},{y},{norm_z}"
            for i, (block, x, y, norm_z) in enumerate(normalized_entries, 1)
        ]

        planner = LegoPlanner(grid_width=6, grid_depth=6, grid_height=3)
        for block in new_blocks:
            planner.add_block(Block(
                id=str(block["id"]),
                color=block["color"],
                block_type=BlockType(block["shape"])
            ))
        for constraint in new_constraints:
            planner.add_absolute_constraint(planner.parse_proposition("D1 "+ constraint))
        solv, plan = planner.solve()
        return solv, plan


    def run_check_solution(self, json):
        """Runs check_solution in a separate thread and stores the result."""
        solv, plan = self.check_solution(json)
        with self.lock:
            self.solution_result = (solv, plan)


    def get_output(
            self,
            actions: DpipActionInterface
    ):
        
        if actions.is_new() and not actions == EmptyInterface():
            start = time.time()
            json = actions.jsonStructure

            check_thread = threading.Thread(target=self.run_check_solution, args=(json,))
            check_thread.start()
            end = time.time()
            # print("planner output took ", end-start)
            with self.lock:
                if self.solution_result is not None:
                    solv, plan = self.solution_result
                    return DpipPlannerInterface(solv, plan)

        return None