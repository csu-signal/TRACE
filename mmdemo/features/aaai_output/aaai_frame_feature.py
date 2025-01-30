from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ColorImageInterface,
    EngagementLevelInterface,
    GazeConesInterface,
    CameraCalibrationInterface
)

from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import camera_3d_to_pixel

lime = (0, 255, 0)
orange = (255, 140, 0)
red = (255, 0, 0)
black = (0, 0, 0)
white = (255, 255, 255)
gray = (128, 128, 128)

@final
class AAAIFrame(BaseFeature[ColorImageInterface]):
    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        gaze: BaseFeature[GazeConesInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        engagement_level: BaseFeature[EngagementLevelInterface],
        draw_cone: bool
    ):
        super().__init__(color, gaze, calibration, engagement_level)
        self.draw_cone = draw_cone

    def get_output(
        self,
        color: ColorImageInterface,
        gaze: GazeConesInterface,
        calibration: CameraCalibrationInterface,
        el: EngagementLevelInterface
    ):
        if not color.is_new():
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)

        #the default image shape is 1080 * 1920 * 3

        #draw white rectangle and its black borderline
        cv.rectangle(output_frame, (30, 260), (205, 500), white, -1, cv.LINE_8)
        cv.rectangle(output_frame, (30, 260), (205, 500), black, 1, cv.LINE_8)

        #print Behavioral Engagement Level on the frame
        cv.putText(output_frame, "Behavioral", (55, 290), cv.FONT_HERSHEY_SIMPLEX, 0.8, black, 1, cv.LINE_AA)
        cv.putText(output_frame, "Engagement", (40, 320), cv.FONT_HERSHEY_SIMPLEX, 0.8, black, 1, cv.LINE_AA)
        cv.putText(output_frame, "Level", (86, 350), cv.FONT_HERSHEY_SIMPLEX, 0.8, black, 1, cv.LINE_AA)

        #draw engagement meter
        cv.rectangle(output_frame, (70, 380), (165, 470), gray, -1, cv.LINE_8)
        if el.engagement_level == 3:
            cv.rectangle(output_frame, (70, 380), (165, 410), lime, -1, cv.LINE_8)
        elif el.engagement_level == 2:
            cv.rectangle(output_frame, (70, 410), (165, 440), orange, -1, cv.LINE_8)
        else:
            cv.rectangle(output_frame, (70, 440), (165, 470), red, -1, cv.LINE_8)

        #draw borderline for engagement meter
        cv.rectangle(output_frame, (70, 380), (165, 410), black, 1, cv.LINE_8)
        cv.rectangle(output_frame, (70, 410), (165, 440), black, 1, cv.LINE_8)
        cv.rectangle(output_frame, (70, 440), (165, 470), black, 1, cv.LINE_8)

        if self.draw_cone:
            for cone in gaze.cones:
                AAAIFrame.projectVectorLines(
                    cone, output_frame, calibration, False, False, True
                )

        output_frame = cv.resize(output_frame, (1280, 720))

        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)
    
    @staticmethod
    def projectVectorLines(cone: Cone, frame, calibration, includeY, includeZ, gaze):
        """
        Draws lines representing a 3d cone onto the frame.

        Arguments:
        cone -- the cone object
        frame -- the frame
        calibration -- the camera calibration settings
        includeY -- a flag to include the Y lines
        includeZ -- a flag to include the Z lines
        gaze -- a flag indicating if we are rendering a gaze vector
        """
        baseUpY, baseDownY, baseUpZ, baseDownZ = AAAIFrame.conePointsBase(cone)
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = AAAIFrame.conePointsVertex(
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