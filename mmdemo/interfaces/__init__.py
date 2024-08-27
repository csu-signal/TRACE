"""
Premade output interfaces
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces.data import Cone, Handedness, ObjectInfo2D, ObjectInfo3D


@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output
    """


@dataclass
class ColorImageInterface(BaseInterface):
    """
    frame_count -- the current frame count
    frame -- image data with shape (h, w, 3) in RGB format.
        The values should be integers between 0 and 255.
    """

    frame_count: int
    frame: np.ndarray


@dataclass
class DepthImageInterface(BaseInterface):
    """
    frame_count -- the current frame count
    frame -- depth image with shape (h, w). The values should
        have type uint16 (integers between 0 and 65535).
    """

    frame_count: int
    frame: np.ndarray


@dataclass
class BodyTrackingInterface(BaseInterface):
    """
    bodies -- [{
                    'body_id': unique identifier,
                    'joint_positions': [xyz position],
                    'joint_orientation': [wxyz quaternion]
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
    camera_matrix: np.ndarray
    distortion: np.ndarray
    """
    rotation: np.ndarray
    translation: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray


@dataclass
class PoseInterface(BaseInterface):
    """
    poses -- list of (participant id, "leaning in" / "leaning out")
    """

    poses: list[tuple[str, str]]


@dataclass
class ConesInterface(BaseInterface):
    # TODO: docstring
    cones: list[Cone]


@dataclass
class GazeConesInterface(ConesInterface):
    # TODO: docstring
    body_ids: list[int]


@dataclass
class GestureConesInterface(ConesInterface):
    # TODO: docstring
    body_ids: list[int]
    handedness: list[Handedness]


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

    objects: Sequence[tuple[ObjectInfo2D | ObjectInfo3D, bool]]


@dataclass
class AudioFileInterface(BaseInterface):
    """
    Audio chunks as a wav file

    `speaker_id` -- unique identifier for the speaker
    `start_time`, `end_time` -- start and end time in seconds
    `path` -- path to wav file
    """

    speaker_id: str
    start_time: float
    end_time: float
    path: Path


@dataclass
class AudioFileListInterface(BaseInterface):
    """
    An interface to return a list of AudioFileInterfces. This
    is useful because sometimes multiple audio files will need
    to be returned in a single frame.

    `audio_files` -- the list of audio files
    """

    audio_files: list[AudioFileInterface]


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
    """
    speaker_id -- the speaker who created the move
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
