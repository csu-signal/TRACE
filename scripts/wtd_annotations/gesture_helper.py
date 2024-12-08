import argparse
import csv
import json
import os
from enum import Enum

import numpy as np

from mmdemo.interfaces.data import GamrTarget


def loadKeyFrameFeatures(csvFiles):
    # Features from the dataset
    featuresArray = []
    for c in csvFiles:
        features = np.loadtxt(
            c, delimiter=",", dtype=str, ndmin=2, usecols=list(range(0, 8)), skiprows=1
        )
        if features.size != 0:
            for f in features:
                featuresArray.append(f)
    return featuresArray


def loadGamrFeatures(csvFile):
    featuresArray = []
    features = np.loadtxt(
        csvFile, delimiter=",", ndmin=2, dtype=str, usecols=list(range(0, 4))
    )
    if features.size != 0:
        for f in features:
            featuresArray.append(f)
        return featuresArray


def convertGamrValues(totalSeconds, gamrFeatures):
    gamrArray = []
    applicableGamr = list(
        filter(
            lambda g: totalSeconds >= float(g[1]) and totalSeconds <= float(g[2]),
            gamrFeatures,
        )
    )
    for app in applicableGamr:
        gamrArray.append(Gamr(app[3]))
    return gamrArray


class GamrCategory(str, Enum):
    UNKNOWN = "unknown"
    EMBLEM = "emblem"
    DEIXIS = "deixis"


class Gamr:
    def __init__(self, string):
        self.string = string
        split = string.split(":")
        self.targets = []

        if GamrCategory.DEIXIS in split[0]:
            self.category = GamrCategory.DEIXIS
        elif GamrCategory.EMBLEM in split[0]:
            self.category = GamrCategory.EMBLEM
        else:
            self.category = GamrCategory.UNKNOWN

        for s in split:
            if "ARG1" in s:
                if GamrTarget.SCALE in s:
                    self.targets.append(GamrTarget.SCALE)
                elif GamrTarget.BLOCKS in s:
                    self.targets.append(GamrTarget.BLOCKS)
                elif GamrTarget.RED_BLOCK in s:
                    self.targets.append(GamrTarget.RED_BLOCK)
                elif GamrTarget.YELLOW_BLOCK in s:
                    self.targets.append(GamrTarget.YELLOW_BLOCK)
                elif GamrTarget.PURPLE_BLOCK in s:
                    self.targets.append(GamrTarget.PURPLE_BLOCK)
                elif GamrTarget.GREEN_BLOCK in s:
                    self.targets.append(GamrTarget.GREEN_BLOCK)
                elif GamrTarget.BLUE_BLOCK in s:
                    self.targets.append(GamrTarget.BLUE_BLOCK)
                elif GamrTarget.MYSTERY_BLOCK in s:
                    self.targets.append(GamrTarget.MYSTERY_BLOCK)
                else:
                    self.targets.append(GamrTarget.UNKNOWN)


def create_gesture_input(gamrPath, annotationsFile, outputFile):
    assert not os.path.exists(outputFile), "output file already exists"
    with open(outputFile, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "blocks", "body_id", "handedness"])

        pointAnnotations = [annotationsFile]
        pointFeatures = loadKeyFrameFeatures(pointAnnotations)
        gamrFeatures = loadGamrFeatures(gamrPath)

        # for each ground truth pointing frame find the ground truth gamrs and log the targets
        for f in pointFeatures:
            start = f[4]
            end = f[5]
            for frameCount in range(int(start), int(end), 1):
                totalSeconds = frameCount / 30  # 30 frames per second
                gamrs = convertGamrValues(totalSeconds, gamrFeatures)

                if gamrs:
                    blocks = []
                    for g in gamrs:
                        if g.category == GamrCategory.DEIXIS:
                            for t in g.targets:
                                # ignore unknowns
                                if (
                                    t != GamrTarget.UNKNOWN
                                    and t != GamrTarget.SCALE
                                    and t != GamrTarget.MYSTERY_BLOCK
                                ):
                                    blocks.append(t.value)

                    if len(blocks) > 0:
                        writer.writerow([frameCount, json.dumps(blocks), f[2], f[1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gamrPath",
        nargs="?",
        default="E:\\Weights_Task\\Data\\GAMR\\CSV\\Group_05_merge_CM.csv",
    )
    # parser.add_argument('--annotationsFile', nargs='?', default="E:\\Weights_Task\\Data\\Pointing\\KeyFrameSelectionOutput\\Group_1-master\\point\\combined.csv")
    parser.add_argument(
        "--annotationsFile",
        nargs="?",
        default="E:\\Weights_Task\\Data\\Pointing\\GroundTruthFrames\\Group5.csv",
    )
    parser.add_argument(
        "--outputFile",
        nargs="?",
        default="E:\\Weights_Task\\Data\\FactPostProcessing\\Gesture\\Group5.csv",
    )
    args = parser.parse_args()

    create_gesture_input(args.gamrPath, args.annotationsFile, args.outputFile)
