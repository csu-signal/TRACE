"""
Premade output interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces.data import ObjectInfo2D, ObjectInfo3D, UtteranceInfo


@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output
    """


class CameraCalibrationInterface(BaseInterface):
    # TODO: brady add docstring
    """
    rotation
    translation: np.ndarray
    cameraMatrix: np.ndarray
    distortion: np.ndarray
    """
    rotation: np.ndarray
    translation: np.ndarray
    cameraMatrix: np.ndarray
    distortion: np.ndarray


@dataclass
class ColorImageInterface(BaseInterface):
    """
    frame -- image data with shape (h, w, 3)
    """

    frame_count: int
    frame: np.ndarray


@dataclass
class DepthImageInterface(BaseInterface):
    """
    frame -- depth image with shape (h, w)
    """

    frame_count: int
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
    """

    bodies: list[dict[str, Any]]
    timestamp_usec: int


@dataclass
class ObjectInterface2D(BaseInterface):
    """
    Object detector outputs

    objects -- list of object locations and classes
    """

    objects: list[ObjectInfo2D]


@dataclass
class ObjectInterface3D(BaseInterface):
    """
    Object detector 3d locations

    objects -- list of object locations and classes
    """

    objects: list[ObjectInfo3D]


@dataclass
class SelectedObjectsInterface(BaseInterface):
    """
    Which objects are selected by participants

    objects -- [(object info, selected?), ...]
    """

    objects: list[tuple[ObjectInfo2D | ObjectInfo3D, bool]]


# TODO: I assume these should be 2d for easier use with
# with rgb-only demos. We could also have separate ones
# for 2d/3d?


@dataclass
class Vectors2D(BaseInterface):
    # TODO: Hannah change if needed
    """
    vectors -- list of numpy vectors with shape (2,)
    """

    vectors: list[np.ndarray]


@dataclass
class Vectors3D(BaseInterface):
    # TODO: Hannah change if needed
    """
    vectors -- list of numpy vectors with shape (3,)
    """

    vectors: list[np.ndarray]


@dataclass
class UtteranceChunkInterface(BaseInterface):
    """
    Audio segments of utterances

    info -- identifying data for the utterance
    audio_file -- path to the audio file
    """

    speaker_id: str
    start_time: float
    end_time: float
    audio_file: Path


@dataclass
class TranscriptionInterface(BaseInterface):
    # TODO: brady figure out how to get frames
    """
    Transcribed utterances

    info -- identifying data for the utterance
    text -- text of the utterance
    """

    speaker_id: str
    start_time: float
    end_time: float
    text: str


@dataclass
class PropositionInterface(BaseInterface):
    # TODO: brady docstring
    """
    prop -- proposition as a string
    """
    speaker_id: str
    prop: str


@dataclass
class MoveInterface(BaseInterface):
    # TODO: brady docstring
    """
    info -- identifying data for the utterance
    move -- iterable containing some subset of
            {"STATEMENT", "ACCEPT", "DOUBT"}
    """
    speaker_id: str
    move: Iterable[str]


@dataclass
class CommonGroundInterface(BaseInterface):
    """
    qbank, fbank, ebank -- sets of propositions describing the
        current common ground
    """

    qbank: set[str]
    fbank: set[str]
    ebank: set[str]


@dataclass
class PoseInterface(BaseInterface):
    """
    poses -- list of (body_id, "leaning in" / "leaning out")
    """

    poses: list[tuple[int, str]]
