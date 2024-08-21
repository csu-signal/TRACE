import json
from enum import Enum

import cv2
import numpy as np


class GamrCategory(str, Enum):
    UNKNOWN = "unknown"
    EMBLEM = "emblem"
    DEIXIS = "deixis"


# TODO: can probably move to interfaces/data.py?
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
