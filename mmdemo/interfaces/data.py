"""
Helper dataclasses for interface definitions
"""

from dataclasses import dataclass


@dataclass
class ObjectInfo2D:
    """
    p1 -- top left?
    p2 -- bottom right?
    object_class -- ?
    """

    # TODO: what are p1 and p2?
    p1: tuple[float, float]
    p2: tuple[float, float]

    object_class: str


@dataclass
class ObjectInfo3D:
    """
    center -- center of block
    object_class -- ?
    """

    center: tuple[float, float, float]

    object_class: str


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
