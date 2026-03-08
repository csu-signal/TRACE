"""
Premade output interfaces
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces.data import (
    Cone,
    DpipObjectInfo2D,
    DpipObjectInfo3D,
    Handedness,
    HciiObjectInfo2D,
    ObjectInfo2D,
    ObjectInfo3D,
    ParticipantInfo,
)


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
    """
    rotation: rotation matrix with shape (3,3)
    translation: translation vector with shape (3,)
    camera_matrix: camera matrix with shape (3,3)
    distortion: distortion values with shape (8,)
    """

    rotation: np.ndarray
    translation: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray


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
    """
    cones -- the list of cones found
    """

    cones: list[Cone]


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
class DpipCommonGroundTrackingInterface(CommonGroundInterface):
    """
    returns
    """


@dataclass
class EmptyInterface(BaseInterface):
    """
    Output interface when the feature does not have any output
    """


@dataclass
class EngagementLevelInterface(BaseInterface):
    engagement_level: int


@dataclass
class FrictionMetrics:
    """Metrics for generated friction statements"""

    nll: float
    predictive_entropy: float
    mutual_information: float
    perplexity: float
    conditional_entropy: float


@dataclass
class DpipFrictionOutputInterface(BaseInterface):
    # TODO add props for board state
    friction_statement: str
    transciption_subset: str
    cg_json: str
    ranking: str


@dataclass
class FrictionOutputInterface(BaseInterface):
    """
    Interface for friction generation output in collaborative weight estimation task.

    Attributes:
        friction_statement (str):
            Main friction statement to be displayed/spoken.
            Example: "Are we sure about comparing these blocks without considering their volume?"

        transciption_subset (str):
            The transcription subset

        task_state (str):
            Current state of the weight estimation task.
            Hidden from UI but useful for debugging.
            Example: "Red (10g) and Blue blocks compared, Yellow block pending"

        belief_state (str):
            Participants' current beliefs about weights.
            Helps explain friction but may not need display.
            Example: "P1 believes yellow is heaviest, P2 uncertain about blue"

        rationale (str):
            Reasoning behind the friction intervention.
            Could be shown as tooltip/explanation.
            Example: "Participants are making assumptions without evidence"

        metrics (Optional[FrictionMetrics]):
            Model's generation metrics including confidence.
            Useful for debugging and demo insights.
    """

    friction_statement: str
    transciption_subset: str
    # task_state: str
    # belief_state: str
    # rationale: str
    # raw_generation: str

    # metrics: Optional[FrictionMetrics] = None

    def to_dict(self):
        return asdict(self)  # Converts the object into a dictionary


@dataclass
class GazeConesInterface(ConesInterface):
    """
    `cones` -- the list of cones found
    `wtd_body_ids` -- `body_ids[i]` is the body id in weight task dataset format for each `cones[i]`
    `azure_body_ids` -- `body_ids[i]` is the body id in azure kinect format for each `cones[i]`
    """

    wtd_body_ids: list[int]
    azure_body_ids: list[int]


@dataclass
class GestureConesInterface(ConesInterface):
    """
    `cones` -- the list of cones found
    `wtd_body_ids` -- `body_ids[i]` is the body id in weight task dataset format for each `cones[i]`
    `azure_body_ids` -- `body_ids[i]` is the body id in azure kinect format for each `cones[i]`
    `handedness` -- `handedness[i]` is the hand used to create `cones[i]`
    """

    wtd_body_ids: list[int]
    azure_body_ids: list[int]
    handedness: list[Handedness]


@dataclass
class GazeEventInterface(BaseInterface):
    """
    The Interface of GazeEvent feature
    There are more than one participant in the vedio
    both positive and negetive event could happen here
    This interface contains both positive and negative
    """

    positive_event: int
    negative_event: int


@dataclass
class GestureEventInterface(BaseInterface):
    """
    The interface of GestureEvent feature
    The model is used to detect who is pointing and what is pointed
    What is pointed is not important in this task
    We only want to who is pointing
    Pointing is a positive signal, here only positive event included
    """

    positive_event: int


@dataclass
class GazeSelectionInterface(BaseInterface):
    """
    Whether the gaze of a participant selects other participants

    selection -- [(gaze cone owner, gaze cone target or None)]
    """

    selection: list[tuple[str, str | None]]


@dataclass
class HciiGestureConesInterface(ConesInterface):
    """
    `cones` -- the list of cones found
    `wtd_body_ids` -- `body_ids[i]` is the body id in weight task dataset format for each `cones[i]`
    `azure_body_ids` -- `body_ids[i]` is the body id in azure kinect format for each `cones[i]`
    `handedness` -- `handedness[i]` is the hand used to create `cones[i]`
    `nose_position` -- the nose position
    """

    wtd_body_ids: list[int]
    azure_body_ids: list[int]
    handedness: list[Handedness]
    nose_positions: list[float]


@dataclass
class HciiSelectedObjectsInterface(BaseInterface):
    """
    Which objects are selected by participants

    objects -- [(object info, selected?, participant id), ...]
    """

    objects: Sequence[tuple[HciiObjectInfo2D, bool, int]]


@dataclass
class MoveInterface(BaseInterface):
    """
    speaker_id -- unique identifier for the speaker
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
class DpipObjectInterface3D(BaseInterface):
    """
    Object detector 3d locations

    xyGrid -- the x y grid
    """

    xyGrid: set[str]
    frame_index: int
    region_frac: float
    norm_point_prompt_grid: np.ndarray
    crop_bounds: Tuple[int, int, int, int]
    boxes: list
    centers: list
    coords: list
    segmentation_masks: list
    labels: dict


@dataclass
class DpipActionInterface(BaseInterface):
    """
    Action interface

    actions -- the actions
    """

    structure: dict
    jsonStructure: dict


@dataclass
class PlannerInterface(BaseInterface):
    """
    solv -- boolean variable to indicate if the problem is solvable
    """

    solv: bool
    plan: str
    fbank: set[str]


# new interfaces created for AAAI demo
@dataclass
class PoseEventInterface(BaseInterface):
    """
    The interface of PoseEvent feature
    Because different students behave in different ways
    both positive and negative could happen here
    """

    positive_event: int
    negative_event: int


@dataclass
class PoseInterface(BaseInterface):
    """
    poses -- list of (participant id, "leaning in" / "leaning out")
    """

    poses: list[tuple[str, str]]


@dataclass
class PropositionInterface(BaseInterface):
    """
    speaker_id -- unique identifier for the speaker
    prop -- proposition expressed by the speaker
    """

    speaker_id: str
    prop: str


@dataclass
class SelectedObjectsInterface(BaseInterface):
    """
    Which objects are selected by participants

    objects -- [(object info, selected?), ...]
    """

    objects: Sequence[
        tuple[ObjectInfo2D | ObjectInfo3D | DpipObjectInfo2D | DpipObjectInfo3D, bool]
    ]


@dataclass
class SelectedParticipantsInterface(BaseInterface):
    """
    Which participants are selected by participants (via gaze)

    participants -- [(ParticipantInfo info, selected?), ...]
    """

    participants: Sequence[tuple[ParticipantInfo, bool]]


@dataclass
class SpeechOutputInterface(BaseInterface):
    """
    speech_output -- indicates if speech was output
    int -- indicates time to display "idea" joe
    """

    speech_output: bool
    length: int


@dataclass
class TranscriptionInterface(BaseInterface):
    """
    speaker_id -- unique identifier for the speaker
    start_time -- start time of the utterance
    end_time -- end time of the utterance
    text -- transcription of the utterance
    """

    speaker_id: str
    start_time: float
    end_time: float
    text: str


@dataclass
class UserInterface(BaseInterface):
    """
    returns nothing
    """
