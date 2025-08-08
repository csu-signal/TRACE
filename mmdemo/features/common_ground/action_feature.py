from typing import final

import cv2 as cv
import numpy as np
import copy

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ColorImageInterface,
    DpipActionInterface,
    DpipObjectInterface3D
)

@final
class DpipActionFeature(BaseFeature[ColorImageInterface]):
    """
    Return action feature

    Input interfaces are `DpipObjectInterface3D`,

    Output interface is `DpipActionInterface`
    """

    def __init__(
        self,
        objects: BaseFeature[DpipObjectInterface3D]
    ):
        super().__init__(objects)

    def initialize(self):
        self.actions = []
        self.frame = None
        self.coords = [f"({i}, {j})" for i in range(3) for j in range(3)]
        self.noneTypes = ["", None, "ns"]

        # Rebuild stack structure for each frame
        self.structure = {coord: [] for coord in self.coords}
        self.prev_frame = [['' for _ in range(3)] for _ in range(3)]

    def charToColor(self, c):
        match c:
            case "b":
                return "blue"
            case "o":
                return "orange"
            case "g":
                return "green"
            case "y":
                return "yellow"
            case "r":
                return "red"  
            case "n":
                return "none"
            case _:
                return "unknown"

    def structToJson(self, dv):
        def pad(v): return copy.deepcopy(v) + [["ns", ""]] * (3 - len(v))

        c0_0 = pad(dv["(0, 0)"])
        c0_1 = pad(dv["(0, 1)"])
        c0_2 = pad(dv["(0, 2)"])
        c1_0 = pad(dv["(1, 0)"])
        c1_2 = pad(dv["(1, 2)"])
        c2_0 = pad(dv["(2, 0)"])
        c2_2 = pad(dv["(2, 2)"])

        def cell(c): return {"color": f"{self.charToColor(c[0])}", "size": 2 if c[1] == 'r' else 1}

        json = {
            "D1": {
                "row_0": [cell(c0_0[0]), cell(c1_0[0]), cell(c2_0[0])],
                "row_1": [cell(c0_0[1]), cell(c1_0[1]), cell(c2_0[1])],
                "row_2": [cell(c0_0[2]), cell(c1_0[2]), cell(c2_0[2])]
            },
            "D2": {
                "row_0": [cell(c0_2[0]), cell(c0_1[0]), cell(c0_0[0])],
                "row_1": [cell(c0_2[1]), cell(c0_1[1]), cell(c0_0[1])],
                "row_2": [cell(c0_2[2]), cell(c0_1[2]), cell(c0_0[2])]
            },
            "D3": {
                "row_0": [cell(c2_2[0]), cell(c1_2[0]), cell(c0_2[0])],
                "row_1": [cell(c2_2[1]), cell(c1_2[1]), cell(c0_2[1])],
                "row_2": [cell(c2_2[2]), cell(c1_2[2]), cell(c0_2[2])]
            }
        }
        return json

    def get_output(
        self,
        objects: DpipObjectInterface3D,
    ):
        if not objects.is_new():
            return None

        self.frame = objects.xyGrid
        updated_coords = set()

        if self.frame is not None and self.prev_frame is not None:
            for i in range(3):
                for j in range(3):
                    coord = f"({i}, {j})"
                    if coord in updated_coords:
                        continue

                    try:
                        prev_val = self.prev_frame[i][j]
                        curr_val = self.frame[i][j]
                    except IndexError:
                        continue

                    # Skip if both empty
                    if prev_val in self.noneTypes and curr_val in self.noneTypes:
                        continue

                    # Handle reappearance of same color at base layer as a removal
                    if (prev_val not in self.noneTypes and curr_val not in self.noneTypes and
                        prev_val != curr_val and curr_val in self.structure[coord]):
                        # A previous top layer block was removed
                        top_block = self.structure[coord][-1] if self.structure[coord] else None
                        if top_block != curr_val:
                            self.structure[coord].remove(top_block)
                            self.actions.append(f"Frame {objects.frame_index}: a {top_block} has been removed at {coord} (restoring {curr_val})")
                            updated_coords.add(coord)
                        continue

                    # Skip if color unchanged (even if shape changed)
                    if prev_val not in self.noneTypes and curr_val not in self.noneTypes:
                        if prev_val[0] == curr_val[0]:
                            continue

                    # Check addition
                    if curr_val not in self.noneTypes:
                        added = False
                        for di, dj in [(0, 1), (1, 0)]:
                            ni, nj = i + di, j + dj
                            ncoord = f"({ni}, {nj})"
                            if 0 <= ni < 3 and 0 <= nj < 3:
                                n_prev_val = self.prev_frame[ni][nj]
                                n_curr_val = self.frame[ni][nj]

                                if n_curr_val == curr_val and curr_val.endswith('r') and n_prev_val in self.noneTypes:
                                    self.actions.append(f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}")
                                    self.actions.append(f"Frame {objects.frame_index}: a {curr_val} has been added at {ncoord}")
                                    self.structure[coord].append(curr_val)
                                    self.structure[ncoord].append(curr_val)
                                    updated_coords.update({coord, ncoord})
                                    added = True
                                    break
                        if not added:
                            self.actions.append(f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}")
                            self.structure[coord].append(curr_val)
                            updated_coords.add(coord)

                    # Check removal
                    elif prev_val not in self.noneTypes and curr_val in self.noneTypes:
                        if prev_val in self.structure[coord]:
                            self.structure[coord].remove(prev_val)
                            self.actions.append(f"Frame {objects.frame_index}: a {prev_val} has been removed at {coord}")
                            updated_coords.add(coord)

                            if prev_val.endswith('r'):
                                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    ni, nj = i + di, j + dj
                                    ncoord = f"({ni}, {nj})"
                                    if 0 <= ni < 3 and 0 <= nj < 3 and prev_val in self.structure[ncoord]:
                                        self.structure[ncoord].remove(prev_val)
                                        self.actions.append(f"Frame {objects.frame_index}: a {prev_val} has been removed at {ncoord}")
                                        updated_coords.add(ncoord)
                                        break

            if updated_coords:
                print("Structure Update: " + str(self.structure) + "\n")
                print("Json Structure: " + str(self.structToJson(self.structure)))

            self.prev_frame = self.frame

        return DpipActionInterface(structure=self.structure, jsonStructure=self.structToJson(self.structure))
