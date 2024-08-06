"""
Helper dataclasses for interface definitions
"""

from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """
    TODO: docstring once the other TODOs are resolved
    """

    # TODO: what are p1 and p2?
    p1: tuple[float, float]
    p2: tuple[float, float]

    # TODO: should this be Gamr, float, str, ...?
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
