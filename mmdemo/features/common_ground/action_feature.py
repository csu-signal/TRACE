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
        c0_0 = copy.deepcopy(dv["(0, 0)"])
        while len(c0_0) < 3:
            c0_0.append("ns") #pad the array with "none sqaures"

        c0_1 = copy.deepcopy(dv["(0, 1)"])
        while len(c0_1) < 3:
            c0_1.append("ns")

        c0_2 = copy.deepcopy(dv["(0, 2)"])
        while len(c0_2) < 3:
            c0_2.append("ns")

        c1_0 = copy.deepcopy(dv["(1, 0)"])
        while len(c1_0) < 3:
            c1_0.append("ns")

        c1_2 = copy.deepcopy(dv["(1, 2)"])
        while len(c1_2) < 3:
            c1_2.append("ns")

        c2_0 = copy.deepcopy(dv["(2, 0)"])
        while len(c2_0) < 3:
            c2_0.append("ns")

        c2_2 = copy.deepcopy(dv["(2, 2)"])
        while len(c2_2) < 3:
            c2_2.append("ns")

        json = {
            "D1": {
                "row_0": [
                    {"color":f"{self.charToColor(c0_0[0][0])}", "size":2 if c0_0[0][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c1_0[0][0])}", "size":2 if c1_0[0][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c2_0[0][0])}", "size":2 if c2_0[0][1] == 'r' else 1},
                    ],
                "row_1": [
                    {"color":f"{self.charToColor(c0_0[1][0])}", "size":2 if c0_0[1][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c1_0[1][0])}", "size":2 if c1_0[1][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c2_0[1][0])}", "size":2 if c2_0[1][1] == 'r' else 1},
                    ],
                "row_2": [
                    {"color":f"{self.charToColor(c0_0[2][0])}", "size":2 if c0_0[2][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c1_0[2][0])}", "size":2 if c1_0[2][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c2_0[2][0])}", "size":2 if c2_0[2][1] == 'r' else 1},
                ]
            },
            "D2": {
                "row_0": [
                    {"color":f"{self.charToColor(c0_2[0][0])}", "size":2 if c0_2[0][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c0_1[0][0])}", "size":2 if c0_1[0][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_0[0][0])}", "size":2 if c0_0[0][1] == 'r' else 1}, 
                    ],
                "row_1": [
                    {"color":f"{self.charToColor(c0_2[1][0])}", "size":2 if c0_2[1][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c0_1[1][0])}", "size":2 if c0_1[1][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_0[1][0])}", "size":2 if c0_0[1][1] == 'r' else 1}, 
                    ],
                "row_2": [
                    {"color":f"{self.charToColor(c0_2[2][0])}", "size":2 if c0_2[2][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c0_1[2][0])}", "size":2 if c0_1[2][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_0[2][0])}", "size":2 if c0_0[2][1] == 'r' else 1}, 
                ]
            },
            "D3": {
                "row_0": [
                    {"color":f"{self.charToColor(c2_2[0][0])}", "size":2 if c2_2[0][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c1_2[0][0])}", "size":2 if c1_2[0][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_2[0][0])}", "size":2 if c0_2[0][1] == 'r' else 1}, 
                    ],
                "row_1": [
                    {"color":f"{self.charToColor(c2_2[1][0])}", "size":2 if c2_2[1][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c1_2[1][0])}", "size":2 if c1_2[1][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_2[1][0])}", "size":2 if c0_2[1][1] == 'r' else 1}, 
                    ],
                "row_2": [
                    {"color":f"{self.charToColor(c2_2[2][0])}", "size":2 if c2_2[2][1] == 'r' else 1},
                    {"color":f"{self.charToColor(c1_2[2][0])}", "size":2 if c1_2[2][1] == 'r' else 1}, 
                    {"color":f"{self.charToColor(c0_2[2][0])}", "size":2 if c0_2[2][1] == 'r' else 1}, 
                ]
            }
        }
        return json
        
        
    def get_output(
        self,
        objects: DpipObjectInterface3D,
    ):
        if (not objects.is_new()):
            return None
        
        self.frame = objects.xyGrid
        update = False

         # Compare with previous frame
        if self.frame is not None and self.prev_frame is not None:
            for i in range(3):
                for j in range(3):
                    coord = f"({i}, {j})"
                    try:
                        prev_val = self.prev_frame[i][j]
                        curr_val = self.frame[i][j]
                    except IndexError:
                        continue

                    if curr_val != prev_val:
                        # If something new appears that's never been here before: PUT
                        if prev_val not in self.noneTypes and curr_val not in self.noneTypes:
                            if curr_val != prev_val:
                                action = f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}"
                                print(action)
                                self.actions.append(action)
                                self.structure[coord].append(curr_val)
                                update = True
                        elif prev_val not in self.noneTypes and curr_val in self.noneTypes:
                            action = f"Frame {objects.frame_index}: a {prev_val} has been removed at {coord}"
                            print(action)
                            self.actions.append(action)
                            self.structure[coord].pop()
                            update = True
                        elif prev_val in self.noneTypes and curr_val not in self.noneTypes:
                            action = f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}"
                            print(action)
                            self.actions.append(action)
                            self.structure[coord].append(curr_val)
                            update = True
            if(update):
                print("Structure Update: " + str(self.structure) + "\n")
                print("Json Structure: " + str(self.structToJson(self.structure)))

            # Update previous frame
            self.prev_frame = self.frame

        return DpipActionInterface(structure = self.structure, jsonStructure = self.structToJson(self.structure))
