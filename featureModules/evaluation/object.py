from collections import defaultdict
from featureModules import ObjectFeature
import csv

from utils import Block, GamrTarget

# from Block definition in utils.py
def str_to_gamr_float(s):
    match s:
        case "red":
            return 0
        case "blue":
            return 3
        case "green":
            return 2
        case "purple":
            return 4
        case "yellow":
            return 1
        case _:
            raise ValueError("string does not correspond to a target")

class ObjectFeatureEval(ObjectFeature):
    def __init__(self, input_dir, log_dir=None):
        self.init_logger(log_dir)

        self.blocks_by_frame = defaultdict(list)

        with open(input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i:j for i,j in zip(keys, row)}
                frame_index = int(data["frame_index"])
                class_gamr = str_to_gamr_float(data["class"])
                p10 = float(data["p10"])
                p11 = float(data["p11"])
                p20 = float(data["p20"])
                p21 = float(data["p21"])
                self.blocks_by_frame[frame_index].append(Block(class_gamr, [p10, p11], [p20, p21]))

        self.current_frame = 0


    def processFrame(self, framergb, frameIndex):
        blocks = []
        while self.current_frame <= frameIndex:
            new_blocks = self.blocks_by_frame[self.current_frame]
            for b in new_blocks:
                self.log_block(self.current_frame, b)

            if len(new_blocks) > 0:
                blocks = new_blocks

            self.current_frame += 1

        return blocks
