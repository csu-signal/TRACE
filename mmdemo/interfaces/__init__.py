"""
Premade output interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces.data import Cone, ObjectInfo2D, ObjectInfo3D


@dataclass
class _AudioChunkBase(BaseInterface):
    """Base interface for audio chunks"""

    speaker_id: str
    start_time: float
    end_time: float


# TODO: remove this? if so then combine _AudioChunkBase with AudioFileInterface
# @dataclass
# class AudioBytesInterface(_AudioChunkBase):
#     """
#     Audio chunks as bytes in wav format
#
#     `speaker_id` -- unique identifier for the speaker
#     `start_time`, `end_time` -- start and end time in seconds
#     `frames` -- audio data bytes
#     `sample_rate` -- sample rate of audio bytes
#     `sample_width` -- sample width of audio bytes
#     `channels` -- channels of audio bytes
#     """
#     frames: bytes
#     sample_rate: int
#     sample_width: int
#     channels: int


@dataclass
class AudioFileInterface(_AudioChunkBase):
    """
    Audio chunks as a wav file

    `speaker_id` -- unique identifier for the speaker
    `start_time`, `end_time` -- start and end time in seconds
    `path` -- path to wav file
    """

    path: Path


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
class CameraCalibrationInterface(BaseInterface):
    # TODO: brady add docstring
    """
    rotation: np.ndarray
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
    frame -- image data with shape (h, w, 3) in RGB format.
        The values should be integers between 0 and 255.
    """

    frame_count: int
    frame: np.ndarray


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
class ConesInterface(BaseInterface):
    cones: list[Cone]


@dataclass
class DepthImageInterface(BaseInterface):
    """
    frame -- depth image with shape (h, w). The values should
        have type uint16 (integers between 0 and 65535).
    """

    frame_count: int
    frame: np.ndarray


@dataclass
class GazeConesInterface(ConesInterface):
    body_ids: list[int]


@dataclass
class GestureConesInterface(ConesInterface):
    body_ids: list[int]
    handedness: list[str]


@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output
    """


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
class PoseInterface(BaseInterface):
    """
    poses -- list of (body_id, "leaning in" / "leaning out")
    """

    poses: list[tuple[int, str]]


@dataclass
class PropositionInterface(BaseInterface):
    # TODO: brady docstring
    """
    prop -- proposition as a string
    """
    speaker_id: str
    prop: str


@dataclass
class SelectedObjectsInterface(BaseInterface):
    """
    Which objects are selected by participants

    objects -- [(object info, selected?), ...]
    """

    objects: list[tuple[ObjectInfo2D | ObjectInfo3D, bool]]


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


# TODO: I assume these should be 2d for easier use with
# with rgb-only demos. We could also have separate ones
# for 2d/3d?


@dataclass
class Vectors2DInterface(BaseInterface):
    # TODO: Hannah change if needed
    """
    vectors -- list of numpy vectors with shape (2,)
    """

    vectors: list[np.ndarray]


@dataclass
class Vectors3DInterface(BaseInterface):
    # TODO: Hannah change if needed
    """
    A data class for storing and managing 3D vectors associated with different bodies.

    Attributes:
        vectors (Dict[str, Tuple[np.ndarray, np.ndarray]]): A dictionary mapping each body identifier to a tuple of two numpy arrays.
            - Each key is a string representing the body identifier.
            - Each value is a tuple containing two 3D numpy arrays: the starting point (`_point`) and the ending point (`end_point`) of the vector.
            - Both numpy arrays have the shape (3,), representing vectors in 3-dimensional space.
    """
    vectors: dict[str, tuple[np.ndarray, np.ndarray]]
