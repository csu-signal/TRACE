import warnings
from pathlib import Path
from typing import final

import joblib
import mediapipe as mp
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.helpers import get_average_hand_pixel, normalize_landmarks, fix_body_id
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    FrictionOutputInterface,
    GestureConesInterface,
)
from mmdemo.interfaces.data import Cone, Handedness
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d
from huggingface_hub import InferenceClient

# - huggingface_hub-0.27.1 #TODO update official yaml


@final
class Friction(BaseFeature[FrictionOutputInterface]):
    """
    Detect friction in group work (client side).

    Input interfaces are TODO add inputs

    Output interface is `FrictionOutputInterface`

    Keyword arguments:
    `model_path` -- the path to the model on hugging face (or None to use the default)
    """
    DEFAULT_MODEL_PATH = "" #TODO is there a default path?

    def __init__(
        self,
        #TODO add inputs
        *,
        model_path: Path | None = None
    ):
        super().__init__() #pass inputs into the base constructor
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path

    def initialize(self):
        #hugging face setup here
        self.client = InferenceClient()

    def get_output(
        self,
        #inputs to send to model go in this method,
    ):
        #TODO check if any of the input features have been updated
        #if not color.is_new() or not depth.is_new() or not bt.is_new():
            #return None

        #TODO send data to valid hugging face location, await response
        response = self.client.post(json={"inputs": "An astronaut riding a horse on the moon."}, model="stabilityai/stable-diffusion-2-1")
        #response.content

        #TODO return model real reponse
        return FrictionOutputInterface(response.content)
