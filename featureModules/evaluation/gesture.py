import csv
import json

from featureModules import GestureFeature
from utils import get_frame_bin


class GestureFeatureEval(GestureFeature):
    def __init__(self, input_dir, log_dir = None):
        self.init_logger(log_dir)

        self.blockCache = {}
        self.gestures_by_frame = {}
        
        with open(input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i:j for i,j in zip(keys, row)}
                print(data["blocks"])
                targets = json.loads(data["blocks"])
                frame = int(data["frame"])
                if len(targets) > 0:
                    self.blockCache[get_frame_bin(frame)] = targets
                self.gestures_by_frame[frame] = targets

        self.current_frame = 0


    def processFrame(self, deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus, frameIndex, includeText):
        while self.current_frame <= frameIndex:
            if self.current_frame in self.gestures_by_frame:
                targets = self.gestures_by_frame[self.current_frame]
                self.log_gesture(self.current_frame, targets, -1, "-1")
            self.current_frame += 1
