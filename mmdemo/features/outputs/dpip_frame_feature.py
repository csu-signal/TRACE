import random
import re
from typing import List, Tuple, final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.dpip_config import *
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
class DpipFrame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the EMNLP Demo

    Input interfaces are `ColorImageInterface`, `DpipObjectInterface3D`,
    `DpipActionInterface`, `DpipFrictionOutputInterface`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        # speechoutput: BaseFeature[SpeechOutputInterface],
        color: BaseFeature[ColorImageInterface],
        # gesture: BaseFeature[GestureConesInterface],
        objects: BaseFeature[DpipObjectInterface3D],
        action: BaseFeature[DpipActionInterface],
        friction: BaseFeature[DpipFrictionOutputInterface],
        # plan: BaseFeature[PlannerInterface] | None = None,
    ):
        # if plan is None:
        #     super().__init__(color, gesture, sel_objects, calibration) # removed gaze
        # else:
        # super().__init__(speechoutput, color, objects, action, friction) # removed gaze
        # super().__init__(speechoutput, color, objects, action, friction)
        super().__init__(color, objects, action, friction)

    def initialize(self):
        self.last_plan = {"text": "", "color": (255, 255, 255)}

    def get_output(
        self,
        # speech: SpeechOutputInterface,
        color: ColorImageInterface,
        # gesture: GestureConesInterface,
        objects: DpipObjectInterface3D,
        actions: DpipActionInterface,
        friction: DpipFrictionOutputInterface,
        # plan: PlannerInterface = None,
    ):
        if not color.is_new() or not objects.is_new() or not friction.is_new():
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)
        h, w, _ = color.frame.shape

        # # render gesture vectors
        # for cone in gesture.cones:
        #     DpipFrame.projectVectorLines(
        #         cone, output_frame, calibration, True, False, False
        #     )

        # # render objects
        # for obj in objects.objects:
        #     c = (0, 255, 0) if obj[1] == True else (0, 0, 255)
        #     block = obj[0]
        #     cv.rectangle(
        #         output_frame,
        #         (int(block.p1[0]), int(block.p1[1])),
        #         (int(block.p2[0]), int(block.p2[1])),
        #         color=c,
        #         thickness=5,
        #     )

        # # render plan
        # # if plan:
        # #     DpipFrame.renderPlan(output_frame, plan, self.last_plan)

        if friction and friction.friction_statement != "":
            frictionStatements = friction.friction_statement.split("\n")
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

            # if(len(frictionStatements) > 1):
            #     #friction includes rational, print it
            #     rstate = frictionStatements[1]
            #     x, y = (50, 110)
            #     text = rstate
            #     font = cv.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.5
            #     font_thickness = 1
            #     text_color_bg = (255,255,255)
            #     text_color =(0,0,0)
            #     text_size, _ = cv.getTextSize(str(text), font, font_scale, font_thickness)
            #     text_w, text_h = text_size
            #     cv.rectangle(output_frame, (x - 5,y - 5), (int(x + text_w + 10), int(y + text_h + 10)), text_color_bg, -1)
            #     cv.putText(output_frame, str(text), (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv.LINE_AA)

            # print friction statement

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

        output_frame = self.grid_overlay(output_frame, objects.boxes)
        output_frame = self.segmentation_masks_overlay(
            output_frame, objects.segmentation_masks, alpha=0.6
        )
        # cv.putText(output_frame, f"[W/S] region_frac = {objects.region_frac:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output_frame = cv.resize(output_frame, (1280, 720))
        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)

    # ========== Overlays ==========

    def segmentation_masks_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        region_frac: float = 1,
        alpha: float = 0.5,
        random_colors: bool = True,
    ) -> np.ndarray:
        def generate_random_color():
            return tuple(random.randint(0, 255) for _ in range(3))

        overlay = image.copy()

        h, w = image.shape[:2]
        region_size = region_frac * min(h, w)
        cell_size = region_size / GRID_SIZE
        mask_size_threshold = 0.5 * cell_size**2

        if not masks:
            return overlay

        for mask in masks:
            #            if (is_mask_square(mask) or is_mask_rectangle(mask)) and not is_mask_too_small(mask, mask_size_threshold):
            color = generate_random_color() if random_colors else (0, 255, 0)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            overlay = cv.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
        return overlay

    def grid_overlay(
        self,
        image: np.ndarray,
        boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        overlay = image.copy()
        for idx, (pt1, pt2) in enumerate(boxes):
            cv.rectangle(overlay, pt1, pt2, color, thickness)
        return overlay

    def point_prompt_overlay(
        image: np.ndarray,
        norm_points: np.ndarray,
        crop_bounds: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 3,
        thickness: int = -1,
    ) -> np.ndarray:
        overlay = image.copy()
        x0, y0, x1, y1 = crop_bounds
        crop_w = x1 - x0
        crop_h = y1 - y0

        for x_norm, y_norm in norm_points:
            x = int(x0 + x_norm * crop_w)
            y = int(y0 + y_norm * crop_h)
            cv.circle(overlay, (x, y), radius, color, thickness)

        return overlay

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

        # if gaze:
        #     yColor = (255, 107, 170)
        #     ZColor = (107, 255, 138)
        #     vectorColor = (255, 107, 170)
        # else:
        # yColor = (255, 255, 0)
        # ZColor = (243, 82, 121)
        # vectorColor = (0, 165, 255)
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
        return ""

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
        cv.putText(
            frame,
            last_plan["text"],
            position,
            font,
            font_scale,
            last_plan["color"],
            thickness,
        )

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
