from typing import final

import cv2
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.dpip_config import GRID_SIZE
from mmdemo.interfaces import ColorImageInterface, DpipObjectInterface3D


@final
class DpipBlockDetectionsFrame(BaseFeature[ColorImageInterface]):
    """
    Return a nice looking fixed-size XY grid of the current block detections

    Input interfaces are `DpipObjectInterface3D`

    Output interface is `ColorImageInterface`
    """

    def __init__(
        self,
        objects: BaseFeature[DpipObjectInterface3D],
    ):
        super().__init__(objects)

    def get_output(
        self,
        objects: DpipObjectInterface3D,
    ):
        if not objects.is_new():
            return None

        output_frame = self.show_grid_detections(objects.labels, GRID_SIZE)
        return ColorImageInterface(frame=output_frame, frame_count=objects.frame_index)

    def show_grid_detections(self, labels: dict, grid_size: int):
        canvas_size = 300
        cell_size = canvas_size // grid_size
        margin = 4

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1

        image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        for i in range(grid_size):
            for j in range(grid_size):
                x0 = j * cell_size
                y0 = i * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                label = labels.get((i, j), "")
                if label:
                    parts = label.split("\n")
                    line1 = parts[0]  # e.g., "red square"
                    line2 = parts[1] if len(parts) > 1 else ""

                    # Parse HSV from second line
                    hsv_vals = [
                        int(s.strip())
                        for s in line2.replace("HSV:", "").split(",")
                        if s.strip().isdigit()
                    ]
                    if len(hsv_vals) == 3:
                        hsv = np.uint8([[hsv_vals]])  # shape (1,1,3)
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
                        cell_color = tuple(int(c) for c in rgb)
                    else:
                        cell_color = (50, 50, 50)
                else:
                    line1 = "—"
                    line2 = ""
                    cell_color = (30, 30, 30)

                # Fill background
                cv2.rectangle(image, (x0, y0), (x1, y1), cell_color, thickness=-1)

                # Text color: choose black or white based on brightness
                brightness = np.mean(cell_color)
                font_color = (0, 0, 0) if brightness > 180 else (255, 255, 255)
                line2_color = (60, 60, 60) if brightness > 180 else (180, 180, 180)

                # Draw green border
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

                text_x = x0 + margin
                text_y0 = y0 + margin + 15
                text_y1 = y0 + margin + 30
                text_y2 = y0 + margin + 45

                cv2.putText(
                    image,
                    f"({i},{j})",
                    (text_x, text_y0),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    line1,
                    (text_x, text_y1),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                if line2:
                    cv2.putText(
                        image,
                        line2,
                        (text_x, text_y2),
                        font,
                        font_scale,
                        line2_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

        return image
