import os
import csv

class Logger:
    """
    Simple logging class that the features can use
    to create output csvs. Also provides
    methods to write to text files and stdout.
    """
    def __init__(self, *, file=None, stdout=False):
        self.file = file
        self.stdout = stdout

    def clear(self):
        if self.file is not None:
            if os.path.exists(self.file):
                os.remove(self.file)

    def write_csv_headers(self, *args):
        self.clear()

        if self.file is not None:
            with open(self.file, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(args)

    def append_csv(self, *args):
        if self.file is not None:
            with open(self.file, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(args)

    def append(self, s):
        if self.file is not None:
            with open(self.file, 'a') as f:
                f.write(s)

        if self.stdout:
            print(s, end="")

