import glob
import os

import numpy as np

# robust keyframe annotation utils (read in features, get csv files)


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


# currently based on the peak frames
def featureFilter(feature, frameNumber):
    if frameNumber >= int(feature[4]) and frameNumber <= int(feature[5]):
        return True
    else:
        return False


# currently based on the peak frames
def bodyFilter(feature, bodyId, hand):
    if bodyId == int(feature[2]) and hand == feature[1]:
        return True
    else:
        return False


def objectAnnotationFilter(key, frameNumber):
    return int(key.split("_")[2].split(".")[0]) == frameNumber


def getKeyFrameCsv(dataset):
    csvFiles = []
    for path, subdir, files in os.walk(dataset):
        for file in glob(os.path.join(path, "*.csv")):
            csvFiles.append(file)
    return csvFiles
