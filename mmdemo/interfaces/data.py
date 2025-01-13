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
    """
    A cone in 3d space. The shape is actually not quite a cone
    because it has a nonzero radius at its vertex.

    base -- the center of the base of the cone
    vertex -- the center of the vertex of the cone
    base_radius -- the radius of the cone at its base
    vertex_radius -- the radius of the cone at its vertex
    """

    base: np.ndarray
    vertex: np.ndarray
    base_radius: float
    vertex_radius: float

@dataclass
class SelectedObjectInfo:
    """
    blockName -- the block name
    wtd_id -- the WTD ids
    """

    blockName: str
    wtd_id: []

@dataclass
class ObjectInfo2D:
    """
    p1 -- top left point (x,y)
    p2 -- bottom right point (x,y)
    object_class -- GamrTarget representing the object
    wtd_id -- the WTD ids
    """

    p1: tuple[float, float]
    p2: tuple[float, float]
    object_class: GamrTarget
    wtd_id: []


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
