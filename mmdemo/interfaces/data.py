"""
Helper dataclasses for interface definitions
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


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


# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
class Handedness(Enum):
    Left = "Right"
    Right = "Left"


@dataclass
class Cone:
    # TODO: docstring
    base: np.ndarray
    vertex: np.ndarray
    base_radius: float
    vertex_radius: float


@dataclass
class ObjectInfo2D:
    """
    p1 -- top left?
    p2 -- bottom right?
    object_class -- GamrTarget representing the object
    """

    # TODO: what are p1 and p2?
    p1: tuple[float, float]
    p2: tuple[float, float]

    object_class: GamrTarget


@dataclass
class ObjectInfo3D(ObjectInfo2D):
    """
    center -- center of block
    p1 -- top left in 2d?
    p2 -- bottom right in 2d?
    object_class -- GamrTarget representing the object
    """

    center: tuple[float, float, float]


@dataclass
class UtteranceInfo:
    """
    utterance_id -- a unique identifier for the utterance
    speaker_id -- a unique identifier for the speaker
    start_time -- the starting unix time of the utterance
    start_time -- the ending unix time of the utterance
    """

    utterance_id: int
    speaker_id: str
    start_time: float
    end_time: float
