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
    SelectedObjectsInterface,
    PlannerInterface,
    FrictionOutputInterface
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
class DpipFrame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the EMNLP Demo

    Input interfaces are `ColorImageInterface`, `GazeConesInterface`,
    `GestureConesInterface`, `SelectedObjectsInterface`,
    `FrictionOutputInterface`, `PlannerInterface`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        #gaze: BaseFeature[GazeConesInterface],
        gesture: BaseFeature[GestureConesInterface],
        sel_objects: BaseFeature[SelectedObjectsInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        #friction: BaseFeature[FrictionOutputInterface],
        #plan: BaseFeature[PlannerInterface] | None = None,
    ):
        # if plan is None:
        #     super().__init__(color, gesture, sel_objects, calibration) # removed gaze
        # else:
        super().__init__(color, gesture, sel_objects, calibration) # removed gaze

    def initialize(self):
        self.last_plan = {"text": "", "color": (255, 255, 255)}
        
    def get_output(
        self,
        color: ColorImageInterface,
        # gaze: GazeConesInterface,
        gesture: GestureConesInterface,
        objects: SelectedObjectsInterface,
        calibration: CameraCalibrationInterface,
        # friction: FrictionOutputInterface,
        # plan: PlannerInterface = None,
    ):
        if (
            not color.is_new()
           # or not gaze.is_new()
            or not gesture.is_new()
            or not objects.is_new()
        ):
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)
        output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

        # render gaze vectors
        # for cone in gaze.cones:
        #     DpipFrame.projectVectorLines(
        #         cone, output_frame, calibration, False, False, True
        #     )

        # render gesture vectors
        for cone in gesture.cones:
            DpipFrame.projectVectorLines(
                cone, output_frame, calibration, True, False, False
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

        # render plan
        # if plan:
        #     DpipFrame.renderPlan(output_frame, plan, self.last_plan)
       
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
        baseUpY, baseDownY, baseUpZ, baseDownZ = DpipFrame.conePointsBase(cone)
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = DpipFrame.conePointsVertex(
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
    def getPropValues(propStrings, match):
        """
        Gets the prop values

        Arguments:
        propStrings -- the prop strings array
        match -- the matching color name

        Returns:
        label -- the prop label
        """
        # TODO update for new props
        # label = []
        # for prop in propStrings:
        #     prop_match = re.match(r"(" + match + r")\s*(=|<|>|!=)\s*(.*)", prop)
        #     if prop_match:
        #         block = prop_match[1]
        #         relation = prop_match[2]
        #         rhs = prop_match[3]
        #         if relation == "<" or relation == ">" or relation == "!=":
        #             label.append(relation + rhs)
        #         else:
        #             label.append(rhs)
        return ''

    @staticmethod
    def renderPlan(frame, plan, last_plan):
        """
        Renders the plan text on the frame. 
        If the plan is None, it renders the last known state.
        """
        # if plan and plan.is_new():
        try:
            # Update the last solvable state
            solv = plan.solv
            text = "Solvable" * solv + "Unsolvable" * (not solv)
            last_plan["text"] = text
            last_plan["color"] = (0, 255, 0) if solv else (0, 0, 255)
        # elif not last_plan.get("text"):
        except:
            # Default state if there's no valid last_plan
            last_plan["text"] = "No Plan yet"
            last_plan["color"] = (255, 255, 255)

        # Render the last known plan state
        position = (50, 1000)  # x=10, y=30 (bottom-left corner, with some padding)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        cv.putText(frame, last_plan["text"], position, font, font_scale, last_plan["color"], thickness)


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
