import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface, GazeConesInterface, GestureConesInterface, HciiGestureConesInterface, SelectedObjectsInterface
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.interfaces.data import SelectedObjectInfo

@final
class HciiLog(BaseFeature[EmptyInterface]):
    """
    Log output to stdout and/or csv files. If logging to csv files,
    headers called "log_frame" and "log_time" will be added to represent
    when the output is happening.

    Input interfaces can be any number of `BaseInterfaces`

    Output interface is `EmptyInterface`

    Keyword arguments:
        stdout -- if the interfaces should be printed
        gesture -- the HCII gesture feature
        csv -- if the interfaces should be saved to csvs
        files -- a list of file names inside of the output directory,
                this should be in the same order as input features
        output_dir -- output directory if logging to files
    """

    def __init__(
        self, gesture, selectedObjects, stdout=False, csv=False, fileName=None, output_dir=None
    ) -> None:
        self.stdout = stdout
        self.csv = csv
        self.fileName = f"{fileName}.csv"
        self._out_dir = output_dir

        super().__init__(gesture, selectedObjects)

    def initialize(self):
        if self.csv:
            # create output directory
            if self._out_dir is not None:
                self.output_dir = Path(self._out_dir)
            else:
                self.output_dir = Path(
                    "logging-output-"
                    + datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
                )
            os.makedirs(self.output_dir, exist_ok=True)

            # create output files
            file = self.output_dir / self.fileName
            assert (
                not file.is_file()
            ), f"A logging file already exists and cannot be overwritten ({file})"
            file.touch()
            self.needs_header = True

        self.frame = 0

    def get_output(self, *args):
        logged_something = False
        if(len(args) == 2):
            gesture = args[0]
            objects = args[1]
            if gesture.is_new() or objects.is_new():
                self.log(gesture, objects)
                logged_something = True

        self.frame += 1
        return EmptyInterface() if logged_something else None

    def log(self, gesture: HciiGestureConesInterface, objects: SelectedObjectsInterface):
        if self.stdout:
            print(f"(frame {self.frame:05})", gesture)

        if self.csv:
            file: Path = self.output_dir / self.fileName
            with open(file, "a", newline="") as f:
                if len(gesture.cones) > 0:
                    writer = csv.writer(f)
                    header_row = ["point_frame_index"]
                    output_row = [self.frame]
                    header_row.append("participant_ids")
                    output_row.append(gesture.wtd_body_ids)
                    header_row.append("nose_positions")
                    output_row.append(gesture.nose_positions)
                    header_row.append("selected_object")
                    selectedObjs = []
                    for o in objects.objects:
                        if o[1]:
                            selectedObjs.append(SelectedObjectInfo(str(o[0].object_class), o[0].wtd_id))
                    output_row.append(selectedObjs)

                    if self.needs_header:
                        writer.writerow(header_row)
                        self.needs_header = False
                    writer.writerow(output_row)
