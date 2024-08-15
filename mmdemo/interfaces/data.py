"""
Helper dataclasses for interface definitions
"""
from dataclasses import dataclass

import numpy as np

from mmdemo.utils.Gamr import GamrTarget


@dataclass
class Cone:
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
