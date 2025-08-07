import random
import re
from typing import List, Tuple, final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.dpip_config import *
from mmdemo.interfaces import ColorImageInterface, DpipObjectInterface3D


@final
class DpipObjectsFrame(BaseFeature[ColorImageInterface]):
    """
    Return a minimal output frame that only displays object-related things for the DPIP FACT demo

    Input interfaces are `ColorImageInterface`, `DpipObjectInterface3D`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        objects: BaseFeature[DpipObjectInterface3D],
    ):
        super().__init__(color, objects)

    def initialize(self):
        pass

    def get_output(
        self,
        color: ColorImageInterface,
        objects: DpipObjectInterface3D,
    ):
        if not color.is_new() or not objects.is_new():
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(color.frame)
        h, w, _ = color.frame.shape

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
        output_frame = self.point_prompt_overlay(
            output_frame, objects.norm_point_prompt_grid, objects.crop_bounds
        )
        cv.putText(
            output_frame,
            f"region_frac = {objects.region_frac:.2f}",
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        output_frame = cv.resize(output_frame, (1280, 720))
        return ColorImageInterface(frame=output_frame, frame_count=color.frame_count)

    # ========== Overlays ==========

    def segmentation_masks_overlay(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        alpha: float = 0.5,
        random_colors: bool = True,
    ) -> np.ndarray:
        def generate_random_color():
            return tuple(random.randint(0, 255) for _ in range(3))

        overlay = image.copy()

        if not masks:
            return overlay

        for mask in masks:
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
        self,
        image: np.ndarray,
        norm_points: np.ndarray,
        crop_bounds: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 3,
        thickness: int = -1,
    ) -> np.ndarray:
        overlay = image.copy()

        if crop_bounds is None or norm_points is None:
            return overlay

        x0, y0, x1, y1 = crop_bounds
        crop_w = x1 - x0
        crop_h = y1 - y0

        for x_norm, y_norm in norm_points:
            x = int(x0 + x_norm * crop_w)
            y = int(y0 + y_norm * crop_h)
            cv.circle(overlay, (x, y), radius, color, thickness)

        return overlay
