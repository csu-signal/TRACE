import re
from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    CommonGroundInterface,
    GazeConesInterface,
    GestureConesInterface,
    SelectedObjectsInterface,
)
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
class EMNLPFrame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the EMNLP Demo

    Input interfaces are `ColorImageInterface`, `GazeConesInterface`,
    `GestureConesInterface`, `SelectedObjectsInterface`,
    `CommonGroundInterface`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        gaze: BaseFeature[GazeConesInterface],
        gesture: BaseFeature[GestureConesInterface],
        sel_objects: BaseFeature[SelectedObjectsInterface],
        common_ground: BaseFeature[CommonGroundInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
    ):
        super().__init__(color, gaze, gesture, sel_objects, common_ground, calibration)

    def initialize(self):
        self.has_cgt_data = False

    def get_output(
        self,
        color: ColorImageInterface,
        gaze: GazeConesInterface,
        gesture: GestureConesInterface,
        objects: SelectedObjectsInterface,
        common: CommonGroundInterface,
        calibration: CameraCalibrationInterface,
    ):
        if (
            not color.is_new()
            or not gaze.is_new()
            or not gesture.is_new()
            or not objects.is_new()
        ):
            return None

        output_frame = np.copy(color.frame)
        output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

        # render gaze vectors
        for cone in gaze.cones:
            self.projectVectorLines(cone, output_frame, calibration, False, False, True)

        # render gesture vectors
        for cone in gesture.cones:
            self.projectVectorLines(cone, output_frame, calibration, True, False, False)

        # render objects
        for obj in objects.objects:
            c = (0, 255, 0) if obj[1] == True else (0, 0, 255)
            block = obj[0]
            cv.rectangle(
                output_frame,
                (int(block.p1[0]), int(block.p1[1])),
                (int(block.p2[0]), int(block.p2[1])),
                color=c,
                thickness=5,
            )

        # render common ground
        if self.has_cgt_data or common.is_new():
            self.has_cgt_data = True
            self.renderBanks(output_frame, 130, 260, "FBank", common.fbank)
            self.renderBanks(output_frame, 130, 130, "EBank", common.ebank)
        else:
            self.renderBanks(output_frame, 130, 260, "FBank", set())
            self.renderBanks(output_frame, 130, 130, "EBank", set())

        output_frame = cv.resize(output_frame, (1280, 720))

        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)

    def projectVectorLines(self, cone, frame, calibration, includeY, includeZ, gaze):
        """
        Projects the vector lines on the frame.

        Arguments:
        cone -- the cone object
        frame -- the frame
        calibration -- the camera calibration settings
        includeY -- a flag to include the Y lines
        includeZ -- a flag to include the Z lines
        gaze -- a flag indicating if we are rendering a gaze vector
        """
        baseUpY, baseDownY, baseUpZ, baseDownZ = cone.conePointsBase()
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = cone.conePointsVertex()

        if gaze:
            yColor = (255, 107, 170)
            ZColor = (107, 255, 138)
            vectorColor = (255, 107, 170)
        else:
            yColor = (255, 255, 0)
            ZColor = (243, 82, 121)
            vectorColor = (0, 165, 255)

        base2D = camera_3d_to_pixel(cone.base, calibration)
        vertex2D = camera_3d_to_pixel(cone.vertex, calibration)
        cv.line(frame, base2D, vertex2D, color=vectorColor, thickness=5)

        if includeY:
            baseUp2DY = camera_3d_to_pixel(baseUpY, calibration)
            baseDown2DY = camera_3d_to_pixel(baseDownY, calibration)
            vertexUp2DY = camera_3d_to_pixel(vertexUpY, calibration)
            vertexDown2DY = camera_3d_to_pixel(vertexDownY, calibration)

            pointUpY = (int(baseUp2DY[0]), int(baseUp2DY[1]))
            pointDownY = (int(baseDown2DY[0]), int(baseDown2DY[1]))

            vertexPointUpY = (int(vertexUp2DY[0]), int(vertexUp2DY[1]))
            vertexPointDownY = (int(vertexDown2DY[0]), int(vertexDown2DY[1]))

            cv.line(frame, vertexPointUpY, pointUpY, color=yColor, thickness=5)
            cv.line(frame, vertexPointDownY, pointDownY, color=yColor, thickness=5)

        if includeZ:
            vertexUp2DZ = camera_3d_to_pixel(vertexUpZ, calibration)
            vertexDown2DZ = camera_3d_to_pixel(vertexDownZ, calibration)
            baseUp2DZ = camera_3d_to_pixel(baseUpZ, calibration)
            baseDown2DZ = camera_3d_to_pixel(baseDownZ, calibration)

            pointUpZ = (int(baseUp2DZ[0]), int(baseUp2DZ[1]))
            pointDownZ = (int(baseDown2DZ[0]), int(baseDown2DZ[1]))

            vertexPointUpZ = (int(vertexUp2DZ[0]), int(vertexUp2DZ[1]))
            vertexPpointDownZ = (int(vertexDown2DZ[0]), int(vertexDown2DZ[1]))

            cv.line(frame, vertexPointUpZ, pointUpZ, color=ZColor, thickness=5)
            cv.line(frame, vertexPpointDownZ, pointDownZ, color=ZColor, thickness=5)

    def getPropValues(self, propStrings, match):
        """
        Gets the prop values

        Arguments:
        propStrings -- the prop strings array
        match -- the matching color name

        Returns:
        label -- the prop label
        """
        label = []
        for prop in propStrings:
            prop_match = re.match(r"(" + match + r")\s*(=|<|>|!=)\s*(.*)", prop)
            if prop_match:
                block = prop_match[1]
                relation = prop_match[2]
                rhs = prop_match[3]
                if relation == "<" or relation == ">" or relation == "!=":
                    label.append(relation + rhs)
                else:
                    label.append(rhs)
        return label

    def renderBanks(self, frame, xSpace, yCord, bankLabel, bankValues):
        """
        Renders the bank blocks

        Arguments:
        frame -- the frame
        xSpace -- the X spacing offset
        yCord -- the Y cord to render
        bankLabel -- the bank label
        bankValues -- the bank values
        """
        blocks = len(colors) + 1
        blockWidth = 112
        blockHeight = 112

        h, w, _ = frame.shape
        start = w - (xSpace * blocks)
        p2 = h - yCord
        (tw, th), _ = cv.getTextSize(bankLabel, cv.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        labelCoords = (
            int(start) - int(tw / 4),
            (int(blockHeight / 2) + int(th / 2)) + p2,
        )
        cv.putText(
            frame, bankLabel, labelCoords, cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3
        )

        for i in range(1, blocks):
            p1 = start + (xSpace * i)
            color = colors[i - 1]
            cv.rectangle(
                frame,
                (p1, p2),
                (p1 + blockWidth, p2 + blockHeight),
                color=color.color,
                thickness=-1,
            )

            labels = self.getPropValues(bankValues, color.name)
            numberLabels = min(len(labels), 5)
            if numberLabels > 0:
                for i, line in enumerate(labels):
                    (tw, th), _ = cv.getTextSize(
                        line,
                        cv.FONT_HERSHEY_SIMPLEX,
                        fontScales[numberLabels - 1],
                        fontThickness[numberLabels - 1],
                    )
                    y = (
                        (int(blockHeight / (numberLabels + 1)) + int(th / 3)) * (i + 1)
                    ) + p2
                    x = (int(blockWidth / 2) - int(tw / 2)) + p1
                    cv.putText(
                        frame,
                        line,
                        (x, y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        fontScales[numberLabels - 1],
                        (0, 0, 0),
                        fontThickness[numberLabels - 1],
                    )
