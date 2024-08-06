"""
Premade output interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces.data import ObjectInfo, UtteranceInfo


@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output
    """


@dataclass
class ColorImageInterface(BaseInterface):
    """
    frame -- image data with shape (h, w, 3)
    """

    frame: np.ndarray


@dataclass
class DepthImageInterface(BaseInterface):
    """
    frame -- depth image with shape (h, w)
    """

    frame: np.ndarray


@dataclass
class BodyTrackingInterface(BaseInterface):
    """
    bodies -- [{
                    'body_id': unique identifier,
                    'joint_positions': [xyz position],
                    'join_orientation': [wxyz quaternion]
                }, ...]
    timestamp_usec -- timestamp in microseconds
    num_bodies -- number of unique bodies
    """

    bodies: dict[str, Any]
    timestamp_usec: int
    num_bodies: int


@dataclass
class ObjectInterface(BaseInterface):
    """
    Object detector outputs

    objects -- list of object locations and classes
    """

    objects: list[ObjectInfo]


@dataclass
class SelectedObjectsInterface(BaseInterface):
    """
    Which objects are selected by participants

    objects -- [(object info, selected?), ...]
    """

    objects: list[tuple[ObjectInfo, bool]]


# TODO: I assume these should be 2d for easier use with
# with rgb-only demos. We could also have separate ones
# for 2d/3d?


@dataclass
class GestureInterface(BaseInterface):
    # TODO: gesture interface
    pass


@dataclass
class GazeInterface(BaseInterface):
    # TODO: gaze interface
    pass


@dataclass
class UtteranceChunkInterface(BaseInterface):
    """
    Audio segments of utterances

    info -- identifying data for the utterance
    audio_file -- path to the audio file
    """

    info: UtteranceInfo
    audio_file: Path


@dataclass
class TranscriptionInterface(BaseInterface):
    """
    Transcribed utterances

    info -- identifying data for the utterance
    text -- text of the utterance
    """

    info: UtteranceInfo
    text: str


@dataclass
class PropositionInterface(BaseInterface):
    """
    info -- identifying data for the utterance
    prop -- proposition as a string
    """

    info: UtteranceInfo
    prop: str


@dataclass
class MoveInterface(BaseInterface):
    """
    info -- identifying data for the utterance
    move -- iterable containing some subset of
            {"STATEMENT", "ACCEPT", "DOUBT"}
    """

    info: UtteranceInfo
    move: Iterable[str]


@dataclass
class CommonGroundInterface(BaseInterface):
    """
    qbank, fbank, ebank -- sets of propositions describing the
        current common ground
    """

    qbank: set[str]
    ebank: set[str]
    fbank: set[str]
