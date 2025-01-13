import re
from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    GazeConesInterface,
    GestureConesInterface,
    SelectedObjectsInterface
)
from mmdemo.features.objects.hcii_selected_objects_feature import HciiSelectedObjects
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
class HCII_IT_Frame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the HCII Individual Tracking Paper

    Input interfaces are `ColorImageInterface`, `GazeConesInterface`,
    `GestureConesInterface`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        gaze: BaseFeature[GazeConesInterface],
        gesture: BaseFeature[GestureConesInterface],
        sel_objects: BaseFeature[HciiSelectedObjects],
        calibration: BaseFeature[CameraCalibrationInterface]
    ):
        super().__init__(color, gaze, gesture, sel_objects, calibration)

    def initialize(self):
        self.has_cgt_data = False
        self.last_plan = {"text": "", "color": (255, 255, 255)}     

    def get_output(
        self,
        color: ColorImageInterface,
        gaze: GazeConesInterface,
        gesture: GestureConesInterface,
        objects: SelectedObjectsInterface,
        calibration: CameraCalibrationInterface
    ):
        if (
            not color.is_new()
            or not gaze.is_new()
            or not gesture.is_new()
            or not objects.is_new()
        ):
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)
        output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

        # render gaze vectors
        for index, cone in enumerate(gaze.cones):
            HCII_IT_Frame.projectVectorLines(
                cone, gaze.azure_body_ids[index], 0, index, output_frame, calibration, False, False, True
            )

        # render gesture vectors
        for index, cone in enumerate(gesture.cones):
            HCII_IT_Frame.projectVectorLines(
                cone, gesture.azure_body_ids[index], gesture.wtd_body_ids[index], index, output_frame, calibration, True, False, False
            )

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
        output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2RGB)

        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)

    @staticmethod
    def projectVectorLines(cone: Cone, azureBodyId, wtdBodyId, coneIndex, frame, calibration, includeY, includeZ, gaze):
        """
        Draws lines representing a 3d cone onto the frame.

        Arguments:
        cone -- the cone object
        azureBodyId -- the azure body id
        wtdBodyId -- the WTD body id
        coneIndex -- the index of the cone in the array
        frame -- the frame
        calibration -- the camera calibration settings
        includeY -- a flag to include the Y lines
        includeZ -- a flag to include the Z lines
        gaze -- a flag indicating if we are rendering a gaze vector
        """
        baseUpY, baseDownY, baseUpZ, baseDownZ = HCII_IT_Frame.conePointsBase(cone)
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = HCII_IT_Frame.conePointsVertex(
            cone
        )

        if gaze:
            yColor = (255, 107, 170)
            ZColor = (107, 255, 138)
            vectorColor = (255, 107, 170)
        else:
            yColor = (255, 255, 0)
            ZColor = (243, 82, 121)
            vectorColor = (0, 165, 255)
            textColor = (255, 0, 255)
            cv.putText(
                frame,
                "Azure Participant: " + str(azureBodyId) + " WTD Participant: " + str(wtdBodyId) + " is pointing.",
                (50, 100 + (50 * coneIndex)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                textColor,
                2,
                cv.LINE_AA,
        )

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

    @staticmethod
    def conePointsBase(cone):
        return (
            [cone.base[0], cone.base[1] + cone.base_radius, cone.base[2]],
            [cone.base[0], cone.base[1] - cone.base_radius, cone.base[2]],
            [cone.base[0], cone.base[1], cone.base[2] + cone.base_radius],
            [cone.base[0], cone.base[1], cone.base[2] - cone.base_radius],
        )

    @staticmethod
    def conePointsVertex(cone):
        return (
            [cone.vertex[0], cone.vertex[1] + cone.vertex_radius, cone.vertex[2]],
            [cone.vertex[0], cone.vertex[1] - cone.vertex_radius, cone.vertex[2]],
            [cone.vertex[0], cone.vertex[1], cone.vertex[2] + cone.vertex_radius],
            [cone.vertex[0], cone.vertex[1], cone.vertex[2] - cone.vertex_radius],
        )
