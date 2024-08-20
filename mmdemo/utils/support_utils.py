from enum import Enum
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from numpy.linalg import norm


def createHeadBoundingBox(xAverage, yAverage, xRange, yRange):
    xMax = xAverage + xRange
    xMin = xAverage - xRange
    yMax = yAverage + yRange
    yMin = yAverage - yRange
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    return [
        int(xMin - (xSpan)),
        int(yMin - (ySpan)),
        int(xMax + (xSpan)),
        int(yMax + (ySpan)),
    ]


class ParseResult(Enum):
    Unknown = (0,)
    Success = (1,)
    Exception = (2,)
    InvalidDepth = (3,)
    NoObjects = (4,)
    NoGamr = 5


class Object:
    def __init__(self, id, threeD):
        self.threeD = threeD
        self.id = id


# BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139, 34, 104)]
dotColors = [(0, 0, 139), (20, 128, 48), (71, 130, 170), (205, 95, 58), (205, 150, 205)]


################################################################################

# random utils


def convertTimestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def get_frame_bin(frame):
    return frame // 30
