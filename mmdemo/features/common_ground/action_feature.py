from typing import final

import cv2 as cv
import numpy as np

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

        # Rebuild stack structure for each frame
        self.structure = {coord: [] for coord in self.coords}
        self.prev_frame = [['' for _ in range(3)] for _ in range(3)]
        
    def get_output(
        self,
        objects: DpipObjectInterface3D,
    ):
        if (not objects.is_new()):
            return None
        
        self.frame = objects.xyGrid

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
                        if prev_val not in ["", None] and curr_val not in ["", None]:
                            if curr_val != prev_val:
                                self.actions.append(f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}")
                                self.structure[coord].append(curr_val)
                        elif prev_val not in ["", None] and curr_val in ["", None]:
                            self.actions.append(f"Frame {objects.frame_index}: a {prev_val} has been removed at {coord}")
                            self.structure[coord].pop()
                        elif prev_val in ["", None] and curr_val not in ["", None]:
                            self.actions.append(f"Frame {objects.frame_index}: a {curr_val} has been added at {coord}")
                            self.structure[coord].append(curr_val)
                        print("Structure Update: " + str(self.structure) )

            # Update previous frame
            self.prev_frame = self.frame

        return DpipActionInterface(structure = self.structure)
