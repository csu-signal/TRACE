import json
from enum import Enum

import cv2
import numpy as np


class GamrCategory(str, Enum):
    UNKNOWN = "unknown"
    EMBLEM = "emblem"
    DEIXIS = "deixis"


class GamrTarget(str, Enum):
    UNKNOWN = "unknown"
    SCALE = "scale"
    RED_BLOCK = "red"
    BLUE_BLOCK = "blue"
    YELLOW_BLOCK = "yellow"
    GREEN_BLOCK = "green"
    PURPLE_BLOCK = "purple"
    BROWN_BLOCK = "brown"
    MYSTERY_BLOCK = "mystery"
    BLOCKS = "blocks"


def gamrFeatureFilter(feature, timestamp):
    if timestamp >= float(feature[1]) and timestamp <= float(feature[2]):
        return True
    else:
        return False


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
        filter(lambda g: (gamrFeatureFilter(g, totalSeconds)), gamrFeatures)
    )
    for app in applicableGamr:
        gamrArray.append(Gamr(app[3]))
    return gamrArray


def singleFrameGamrRecall(frame, gamr, blocks, path):
    falseNegative = 0
    falsePositive = 0
    trueNegative = 0
    truePositive = 0
    # for gamr in gamrs:
    if gamr.category != GamrCategory.DEIXIS:
        return False, 0, 0, 0

    blockDescriptions = []

    # ignore unknowns
    if GamrTarget.UNKNOWN in gamr.targets:
        return False, 0, 0, 0
    # if the target is the scale no blocks should be selected (true negative), every selected block is a false positive
    if GamrTarget.SCALE in gamr.targets:
        return False, 0, 0, 0
        # if(len(blocks) > 0):
        #     falsePositive += len(blocks)
        # else:
        #     trueNegative += 1

        # for b in blocks:
        #     blockDescriptions.append(b.description)

    # if the target is the "blocks" each selected block greater than one is a true positive, if none are selected it's a false negative
    elif GamrTarget.BLOCKS in gamr.targets:
        if len(blocks) > 1:
            truePositive += len(blocks)
        else:
            falseNegative += 1

        for b in blocks:
            blockDescriptions.append(b.description)

    # for each block in the gamr if the description matches the gamr target it's a true positive, else it's a false positive
    # if no blocks are selected it's a false negative
    else:
        if len(blocks) > 0:
            for b in blocks:
                blockDescriptions.append(b.description)
                # IOU (interestion over union) jaccard index, dice's coeff, over samples
                if b.description in gamr.targets:
                    truePositive += 1
                else:
                    falsePositive += 1
            # if a block is a gamr target but not in the block list, it's a false negative
            for t in gamr.targets:
                if t not in blockDescriptions:
                    falseNegative += 1

        else:
            for t in gamr.targets:
                falseNegative += 1

    cv2.putText(
        frame,
        "False+: " + str(falsePositive),
        (50, 150),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "True+: " + str(truePositive),
        (50, 200),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "False-: " + str(falseNegative),
        (50, 250),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "True-: " + str(trueNegative),
        (50, 300),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if truePositive > 0 and (falseNegative > 0 or falsePositive > 0):
        precision = round(truePositive / (truePositive + falsePositive), 2)
        recall = round(truePositive / (truePositive + falseNegative), 2)
        f1 = round(2 * (precision * recall) / (precision + recall), 2)

        # return recall
        return True, recall, precision, f1
    return True, 0, 0, 0  # if true postitives are 0 is recall/precision 0?


class GamrStats:
    def __init__(self):
        self.falsePositive = 0
        self.truePositive = 0
        self.falseNegative = 0
        self.trueNegative = 0
        self.totalGamr = 0
        self.failedParse = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.totalIou = 0
        self.averageIou = 0
        self.totalIouFrames = 0

    def analyzeGamr(self, frameIndex, gamr, blocks, path):
        if gamr.category != GamrCategory.DEIXIS:
            return

        dictionary = {
            "frameNumber": frameIndex,
            "category": gamr.category,
            "targets": gamr.targets,
            "blocks": [],
            "IOU": 0,
        }

        blockDescriptions = []
        self.totalGamr += 1
        # print("Target: " + gamr.target)
        # ignore unknowns
        if GamrTarget.UNKNOWN in gamr.targets:
            return
        # if the target is the scale no blocks should be selected (true negative), every selected block is a false positive
        if GamrTarget.SCALE in gamr.targets:
            if len(blocks) > 0:
                self.falsePositive += len(blocks)
            else:
                self.trueNegative += 1

            for b in blocks:
                dictionary["blocks"].append(b.toJSON())
                blockDescriptions.append(b.description)

        # if the target is the "blocks" each selected block greater than one is a true positive, if none are selected it's a false negative
        elif GamrTarget.BLOCKS in gamr.targets:
            if len(blocks) > 1:
                self.truePositive += len(blocks)
            else:
                self.falseNegative += 1

            for b in blocks:
                dictionary["blocks"].append(b.toJSON())
                blockDescriptions.append(b.description)

        # for each block in the gamr if the description matches the gamr target it's a true positive, else it's a false positive
        # if no blocks are selected it's a false negative
        else:
            self.totalIouFrames += 1
            if len(blocks) > 0:
                for b in blocks:
                    dictionary["blocks"].append(b.toJSON())
                    blockDescriptions.append(b.description)
                    # IOU (interestion over union) jaccard index, dice's coeff, over samples
                    if b.description in gamr.targets:
                        self.truePositive += 1
                    else:
                        self.falsePositive += 1
                # if a block is a gamr target but not in the block list, it's a false negative
                for t in gamr.targets:
                    if t not in blockDescriptions:
                        self.falseNegative += 1

            else:
                for t in gamr.targets:
                    falseNegative += 1

        intersection = list(np.intersect1d(blockDescriptions, gamr.targets))
        union = list(np.union1d(blockDescriptions, gamr.targets))
        iou = len(intersection) / len(union)
        dictionary["IOU"] = iou
        self.totalIou += iou
        if self.totalIou > 0:
            self.averageIou = round(self.totalIou / self.totalIouFrames, 2)

        with open(path, "a") as f:
            f.write(json.dumps(dictionary) + ",")
            f.close()

        if self.truePositive > 0 and (self.falseNegative > 0 or self.falsePositive > 0):
            self.precision = round(
                self.truePositive / (self.truePositive + self.falsePositive), 2
            )
            self.recall = round(
                self.truePositive / (self.truePositive + self.falseNegative), 2
            )
            self.f1 = round(
                2 * (self.precision * self.recall) / (self.precision + self.recall), 2
            )


class BlockEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Block:
    def __init__(self, description, p1, p2):
        self.p1 = p1
        self.p2 = p2
        # self.height = height
        # self.width = width
        self.target = False

        if description == 0:
            self.description = GamrTarget.RED_BLOCK

        if description == 1:
            self.description = GamrTarget.YELLOW_BLOCK

        if description == 2:
            self.description = GamrTarget.GREEN_BLOCK

        if description == 3:
            self.description = GamrTarget.BLUE_BLOCK

        if description == 4:
            self.description = GamrTarget.PURPLE_BLOCK

        if description == 5:
            self.description = GamrTarget.SCALE

    def toJSON(self):
        return {
            "description": self.description,
            "x": self.x,
            "y": self.y,
            "height": self.height,
            "width": self.width,
            "target": self.target,
        }


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
