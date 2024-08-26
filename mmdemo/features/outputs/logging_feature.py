import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface


@final
class Log(BaseFeature[EmptyInterface]):
    """
    Log output to stdout and/or csv files. If logging to csv files,
    headers called "log_frame" and "log_time" will be added to represent
    when the output is happening.

    Input interfaces can be any number of `BaseInterfaces`

    Output interface is `EmptyInterface`

    Keyword arguments:
        stdout -- if the interfaces should be printed
        csv -- if the interfaces should be saved to csvs
        files -- a list of file names inside of the output directory,
                this should be in the same order as input features
        output_dir -- output directory if logging to files
    """

    def __init__(
        self, *args, stdout=False, csv=False, files=None, output_dir=None
    ) -> None:
        self.stdout = stdout
        self.csv = csv

        if files is not None:
            self.files = files
            assert len(self.files) == len(
                args
            ), "There should be a file for each input feature"
        else:
            counters = defaultdict(int)
            self.files = []
            for i in args:
                name = i.__class__.__name__
                self.files.append(f"{name}{counters[name]}.csv")
                counters[name] += 1

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(
                "logging-output-"
                + datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
            )

        super().__init__(*args)

    def initialize(self):
        if self.csv:
            os.makedirs(self.output_dir, exist_ok=True)
            for f in self.files:
                file = self.output_dir / f
                assert (
                    not file.is_file()
                ), f"A logging file already exists and cannot be overwritten ({file})"
                file.touch()
            self.needs_header = [True for _ in self.files]

        self.frame = 0

    def get_output(self, *args):
        logged_something = False
        log_time = time.time()
        for index, interface in enumerate(args):
            if not interface.is_new():
                continue

            self.log(interface, index, log_time)
            logged_something = True

        self.frame += 1
        return EmptyInterface() if logged_something else None

    def log(self, interface: BaseInterface, index, log_time):
        if self.stdout:
            print(f"(frame {self.frame:05})", interface)

        if self.csv:
            file: Path = self.output_dir / self.files[index]
            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                header_row = ["log_frame", "log_time"]
                output_row = [self.frame, log_time]
                for field in interface.__dataclass_fields__:
                    if field == "_new":
                        continue
                    header_row.append(field)
                    output_row.append(interface.__getattribute__(field))

                if self.needs_header[index]:
                    writer.writerow(header_row)
                    self.needs_header[index] = False
                writer.writerow(output_row)
