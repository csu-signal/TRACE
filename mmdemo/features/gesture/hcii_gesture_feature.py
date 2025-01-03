import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import get_average_hand_pixel, get_joint, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    HciiGestureConesInterface,
)
from mmdemo.interfaces.data import Cone, Handedness
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d
from mmdemo.utils.joints import Joint


@final
class HciiGesture(BaseFeature[HciiGestureConesInterface]):
    """
    Detect when and where participants are pointing.

    Input interfaces are `ColorImageInterface`, `DepthImageInterface`,
    `BodyTrackingInterface`, `CameraCalibrationInterface`

    Output interface is `HciiGestureConesInterface`

    Keyword arguments:
    `model_path` -- the path to the model (or None to use the default)
    """

    BASE_RADIUS = 40
    VERTEX_RADIUS = 70

    # the number of finger lengths of the output cone
    CONE_FINGER_LENGTHS = 5

    HAND_BOUNDING_BOX_WIDTH = 192
    HAND_BOUNDING_BOX_HEIGHT = 192

    DEFAULT_MODEL_PATH = Path(__file__).parent / "bestModel-pointing.pkl"

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        bt: BaseFeature[BodyTrackingInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        *,
        model_path: Path | None = None
    ):
        super().__init__(color, depth, bt, calibration)
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path
        self.body_idx = []
        self.bt = None

    def initialize(self):
        self.loaded_model = joblib.load(str(self.model_path))

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            static_image_mode=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0,
        )

    def get_output(
        self,
        color: ColorImageInterface,
        depth: DepthImageInterface,
        bt: BodyTrackingInterface,
        calibration: CameraCalibrationInterface,
    ):
        if not color.is_new() or not depth.is_new() or not bt.is_new():
            return None

        cones_output = []
        azure_body_ids_output = []
        wtd_body_ids_output = []
        nose_position_output = []
        handedness_output = []
        bt = fix_body_id(bt)
        for _, body in enumerate(bt.bodies):
            for handedness in (Handedness.Left, Handedness.Right):
                # loop through both hands of all bodies

                # create a box around the hand using azure kinect info
                avg = get_average_hand_pixel(body, calibration, handedness)
                offset = (
                    np.array(
                        [self.HAND_BOUNDING_BOX_WIDTH, self.HAND_BOUNDING_BOX_HEIGHT]
                    )
                    / 2
                )
                box = np.array([avg - offset, avg + offset], dtype=np.int64)
                if (
                    box[0][0] < 0
                    or box[0][1] < 0
                    or box[1][0] >= color.frame.shape[1]
                    or box[1][1] >= color.frame.shape[0]
                ):
                    return None

                # see if the hand is pointing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pointing_info = self.find_pointing_hands(color, handedness, box)

                if pointing_info is None:
                    continue
                base, tip = pointing_info

                try:
                    # calculate 3d cone if the hand is pointing
                    base3D = pixel_to_camera_3d(base, depth, calibration)
                    tip3D = pixel_to_camera_3d(tip, depth, calibration)

                    finger_length = tip3D - base3D

                    cones_output.append(
                        Cone(
                            base3D,
                            base3D + finger_length * self.CONE_FINGER_LENGTHS,
                            self.BASE_RADIUS,
                            self.VERTEX_RADIUS,
                        )
                    )
                    wtd_body_ids_output.append(body["wtd_body_id"])
                    azure_body_ids_output.append(body["body_id"])
                    handedness_output.append(handedness)
                    nose_position_output.append(get_joint(Joint.NOSE, body, calibration))

                except CoordinateConversionError:
                    pass

        return HciiGestureConesInterface(
            wtd_body_ids=wtd_body_ids_output, 
            azure_body_ids=azure_body_ids_output, 
            handedness=handedness_output, 
            cones=cones_output,
            nose_positions=nose_position_output
        )

    def find_pointing_hands(
        self,
        color: ColorImageInterface,
        handedness: Handedness,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Find points inside of a given box of the frame.

        Arguments:
        color -- color image interface
        handedness -- Handedness.Left or Handedness.Right
        box -- [upper left, lower right] where both points are (x,y)

        Returns:
        base -- pixel location of base of index finger
        tip -- pixel location of tip of index finger
        """
        # subframe containing only the hand
        hand_frame = color.frame[box[0][1] : box[1][1], box[0][0] : box[1][0]]

        # run mediapipe
        mediapipe_results = self.hands.process(hand_frame)

        # if we don't have results for hand landmarks, then exit
        if not mediapipe_results.multi_hand_landmarks:
            return None

        # loop through detected hands
        for handedness_value, hand_landmarks in zip(
            mediapipe_results.multi_handedness,
            mediapipe_results.multi_hand_landmarks,
        ):
            # only look at detected hands that match the handedness input
            if handedness_value.classification[0].label != handedness.value:
                continue

            # predict if the hand is pointing and exit if not
            normalized = normalize_landmarks(
                hand_landmarks, hand_frame.shape[1], hand_frame.shape[0]
            )
            prediction = self.loaded_model.predict_proba([normalized])
            if prediction[0][0] < 0.3:
                continue

            # finger landmarks
            mcp_lm = hand_landmarks.landmark[5]
            tip_lm = hand_landmarks.landmark[8]

            # mediapipe outputs values between 0 and 1 so we need to scale the landmark
            # (w, h)
            scale_factor = np.array([hand_frame.shape[1], hand_frame.shape[0]])

            mcp_joint = np.array([mcp_lm.x, mcp_lm.y]) * scale_factor + box[0]
            tip_joint = np.array([tip_lm.x, tip_lm.y]) * scale_factor + box[0]

            return mcp_joint.astype(np.int64), tip_joint.astype(np.int64)
