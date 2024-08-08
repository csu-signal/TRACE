import csv
import os
import shutil

import numpy as np


def initalizeRadiusCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frameIndex",
                "nearRadius",
                "farRadius",
                "recall",
                "precision",
                "f1",
                "targets",
            ]
        )
    return


def LogRadiusCsv(path, frameIndex, nearRadius, farRadius, recall, precision, f1, gamr):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [frameIndex, nearRadius, farRadius, recall, precision, f1, gamr.targets]
        )
    return


def LogAverages(path, label, key, average):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, key, average])
    return


def initalizeCsv(outputFolder, path):
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    os.makedirs(outputFolder)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase_type",
                "hand_index",
                "gesture_type",
                "group",
                "participant",
                "landmarks",
            ]
        )
    return


def LogCsv(
    path, handIndex, gestureType, gesturePhase, group, participant, landmark_list
):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [gesturePhase, handIndex, gestureType, group, participant, *landmark_list]
        )
    return


def initalizeGamrFile(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def loadUtteranceFeatures(csvFile):
    featuresArray = []
    features = np.loadtxt(
        csvFile, delimiter=",", ndmin=2, dtype=str, usecols=list(range(0, 4))
    )
    if features.size != 0:
        for f in features:
            featuresArray.append(f)
        return featuresArray
