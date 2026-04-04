import random
import re
from typing import List, Tuple, final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.friction.sensor_sheet_friction_feature import SensorSheetFrictionFeature
from mmdemo.features.objects.dpip_config import *
from mmdemo.features.speech_output.sensorSpeechoutput_feature import extract_speakable_friction_text
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    DpipActionInterface,
    DpipFrictionOutputInterface,
    DpipObjectInterface3D,
    FrictionOutputInterface,
    GazeConesInterface,
    GestureConesInterface,
    PlannerInterface,
    SelectedObjectsInterface,
    SensorSheetFrictionOutputInterface,
    SpeechOutputInterface,
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import camera_3d_to_pixel


class Color:
    def __init__(self, name, color):
        self.name = name
        self.color = color


colors = [
    Color("red", (0, 0, 255)),
    Color("blue", (255, 0, 0)),
    Color("green", (19, 129, 51)),
    Color("purple", (128, 0, 128)),
    Color("yellow", (0, 215, 255)),
]

fontScales = [1.5, 1.5, 0.75, 0.5, 0.5]
fontThickness = [3, 3, 2, 2, 2]


@final
class SensorFrame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the Sensor Demo

    Input interfaces are `ColorImageInterface`, `DpipObjectInterface3D`,
    `DpipActionInterface`, `DpipFrictionOutputInterface`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        speechoutput: BaseFeature[SpeechOutputInterface],
        color: BaseFeature[ColorImageInterface],
        friction: BaseFeature[SensorSheetFrictionOutputInterface],
    ):
        super().__init__(speechoutput, color, friction)

    def initialize(self):
        self.last_plan = {"text": "", "color": (255, 255, 255)}

    def get_output(
        self,
        speech: SpeechOutputInterface,
        color: ColorImageInterface,
        friction: SensorSheetFrictionOutputInterface,
    ):
        if not color.is_new() or not friction.is_new():
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)
        h, w, _ = color.frame.shape

        if friction and friction.friction_statement != "":
            friction_text = extract_speakable_friction_text(friction.friction_statement)
            frictionStatements = [line for line in friction_text.split("\n") if line.strip()]
            for index, fstate in enumerate(frictionStatements):
                x, y = (50, 75 + (30 * index))
                text = fstate
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                font_thickness = 1
                text_color_bg = (255, 255, 255)
                text_color = (0, 0, 0)
                text_size, _ = cv.getTextSize(
                    str(text), font, font_scale, font_thickness
                )
                text_w, text_h = text_size
                cv.rectangle(
                    output_frame,
                    (x - 5, y - 5),
                    (int(x + text_w + 10), int(y + text_h + 10)),
                    text_color_bg,
                    -1,
                )
                cv.putText(
                    output_frame,
                    str(text),
                    (int(x), int(y + text_h + font_scale - 1)),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv.LINE_AA,
                )

        # draw frame count
        cv.putText(
            output_frame,
            "FRAME:" + str(color.frame_count),
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        output_frame = cv.resize(output_frame, (1280, 720))
        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)