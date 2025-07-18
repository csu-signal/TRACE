from pathlib import Path
from typing import final

import cv2
import numpy as np
import torch

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.config import CLASSES, DEVICE, NUM_CLASSES
from mmdemo.features.objects.model import create_model
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    DpipObjectInterface3D,
)
from mmdemo.interfaces.data import DpipGamrTarget, DpipObjectInfo3D
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d

import time
from typing import List, Tuple
import random

import torch
import cv2
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

GRID_SIZE = 3
DEFAULT_REGION_FRAC = 0.5


@final
class DpipObject(BaseFeature[DpipObjectInterface3D]):
    """
    A feature to get and track the objects through a scene.

    Input interfaces are `ColorImageInterface, DepthImageInterface, CameraCalibrationInterface'.

    Output interface is `DpipObjectInterface3D`.

    Keyword arguments:
    `detection_threshold` -- confidence threshold for the object detector model
    `model_path` -- the path to the model (or None to use the default)
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        *,
        detection_threshold=0.6,
        model_path: Path | None = None,
        skipPost: bool=False
    ) -> None:
        super().__init__(color, depth, calibration)
        self.all_grid_states = {}
        self.skipPost = skipPost
        # self.detectionThreshold = detection_threshold
        # if model_path is None:
        #     self.model_path = self.DEFAULT_MODEL_PATH
        # else:
        #     self.model_path = model_path

    def initialize(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cuda":
            # use bfloat16 (based on the automatic_mask_generator_example.ipynb in the sam2 repo)
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def get_output(
        self,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        calibration: CameraCalibrationInterface,
    ) -> DpipObjectInterface3D | None:
        if(self.skipPost):
            return DpipObjectInterface3D(xyGrid=[], overlayFrame=col.frame) #just to test post transcriptions without a lag
    
        if not col.is_new() or not dep.is_new():
            return None
        
        h, w, _ = col.frame.shape
        region_frac = DEFAULT_REGION_FRAC
        sam2_mask_generator = self.create_sam2_mask_generator()

        segmentation_masks = self.get_segmentation_masks(sam2_mask_generator, col.frame, region_frac)

        boxes, centers, coords = self.build_centered_grid_boxes(col.frame.shape, GRID_SIZE, region_frac)
        labels = self.compute_grid_labels(col.frame, segmentation_masks, boxes)

        xy_grid = self.grid_labels_to_xy_matrix(labels, GRID_SIZE)
        self.all_grid_states[col.frame_count] = xy_grid

        overlay = self.draw_grid_overlay(col.frame, boxes, labels=labels, centers=centers, coords=coords)
        overlay = self.visualize_segmentation_masks(overlay, segmentation_masks, alpha=0.6)

        #TODO move to output feature
        cv2.putText(overlay, f"Frame {col.frame_count}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(overlay, f"[W/S] region_frac = {region_frac:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return DpipObjectInterface3D(xyGrid=xy_grid, overlayFrame = overlay)

    def build_centered_norm_point_grid(self, n_per_side: int, frac: float = 0.5) -> np.ndarray:
        assert 0 < frac <= 1
        pts = np.linspace((1 - frac) / 2, (1 + frac) / 2, n_per_side)
        xs, ys = np.meshgrid(pts, pts)
        return np.stack([xs.ravel(), ys.ravel()], axis=-1)


    def create_sam2_mask_generator(self):
        points_per_axis = 7 
        region_frac = 1 
        centered_grid = self.build_centered_norm_point_grid(points_per_axis, frac=region_frac)

        sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
        sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side = None,
                point_grids = [centered_grid],
        )

        return sam2_mask_generator


    def get_segmentation_masks(self, sam2_mask_generator, image: np.ndarray, region_frac: float):
        try:
            start_time = time.time()

            H, W = image.shape[:2]

            region_size = region_frac * min(H, W)
            cy, cx = H // 2, W // 2
            half = int(region_size / 2)
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            cropped_image = image[y0:y1, x0:x1]

            blurred_cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 1)
            # Bilateral filter supposedly preserves edges
            #blurred_cropped_image = cv2.bilateralFilter(cropped_image, 9, 75, 75)

            sam2_output = sam2_mask_generator.generate(blurred_cropped_image)
            segmentation_masks = [mask["segmentation"] for mask in sam2_output]

            aligned_masks = []
            for mask in segmentation_masks:
                full_mask = np.zeros((H, W), dtype=mask.dtype)
                full_mask[y0:y1, x0:x1] = mask
                aligned_masks.append(full_mask)

            end_time = time.time()
            print(f"time to compute segmentation masks: {end_time - start_time:.4f} seconds")

            return aligned_masks

        except Exception as e:
                print(f"Exception while generating segmentation masks: {e}")


    def grid_labels_to_xy_matrix(self, labels: dict, grid_size: int) -> list[list[str]]:
        grid_matrix = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                label = labels.get((i, j), "")
                if label:
                    # Only keep the first line: "color shape"
                    label = label.split("\n")[0]
                else:
                    label = "unknown"
                # Just get the first letter from the color and shape respectively
                if len(label.split(" ")) == 2:
                    row.append(f"{label.split(' ')[0][0]}{label.split(' ')[1][0]}")
            grid_matrix.append(row)
        return grid_matrix


    def visualize_segmentation_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        alpha: float = 0.5,
        random_colors: bool = True
    ) -> np.ndarray:
        overlay = image.copy()

        # Default color palette
        def generate_color():
            return tuple(random.randint(0, 255) for _ in range(3))

        for mask in masks:
            color = generate_color() if random_colors else (0, 255, 0)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

        return overlay


    def build_centered_grid_boxes(self, image_shape, grid_size, region_frac):
        h, w = image_shape[:2]
        region_size = region_frac * min(h, w)
        cell_size = region_size / grid_size
        cy, cx = h / 2, w / 2
        boxes = []
        centers = []
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                x_center = cx - region_size / 2 + (j + 0.5) * cell_size
                y_center = cy - region_size / 2 + (i + 0.5) * cell_size
                x0 = int(x_center - cell_size / 2)
                y0 = int(y_center - cell_size / 2)
                x1 = int(x_center + cell_size / 2)
                y1 = int(y_center + cell_size / 2)
                boxes.append(((x0, y0), (x1, y1)))
                centers.append((int(x_center), int(y_center)))
                coords.append((i, j))
        return boxes, centers, coords


    def draw_grid_overlay(self, image, boxes, labels=None, centers=None, coords=None, color=(0, 255, 0), thickness=2):
        overlay = image.copy()
        for idx, (pt1, pt2) in enumerate(boxes):
            cv2.rectangle(overlay, pt1, pt2, color, thickness)
            if centers and coords:
                label = f"{coords[idx]}"
                if labels:
                    user_label = labels.get(coords[idx], "")
                    if user_label:
                        label += f"\n{user_label}"
                cx, cy = centers[idx]
                for k, line in enumerate(label.split("\n")):
                    cv2.putText(overlay, line, (cx - 50, cy + k * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return overlay


    def circular_mean_hue(self, hue_values: np.ndarray) -> float:
        angles = np.deg2rad(hue_values * 2)  # Map to [0, 360] → radians
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        mean_angle = np.arctan2(sin_sum, cos_sum)
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return np.rad2deg(mean_angle) / 2  # Back to [0, 179]


    def mean_hsv_from_mask(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        masked_pixels = hsv[mask > 0]

        if masked_pixels.shape[0] == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        mean_h = self.circular_mean_hue(masked_pixels[:, 0])
        mean_s = masked_pixels[:, 1].mean()
        mean_v = masked_pixels[:, 2].mean()

        return np.array([mean_h, mean_s, mean_v], dtype=np.float32)


    def estimate_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[str, Tuple[float, float, float]]:
        mean_hsv = self.mean_hsv_from_mask(image, mask)
        mean_hue = mean_hsv[0]

        if mean_hue < 10 or mean_hue >= 160:
            color_name = "red"
        elif 10 <= mean_hue < 20:
            color_name = "orange"
        elif 20 <= mean_hue < 35:
            color_name = "yellow"
        elif 35 <= mean_hue < 85:
            color_name = "green"
        elif 85 <= mean_hue < 170:
            color_name = "blue"
        else:
            color_name = "unknown"

        return color_name, mean_hsv

    def estimate_shape(self, mask: np.ndarray) -> str:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "unknown"
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if 0.8 < ratio < 1.2:
            return "square"
        elif ratio >= 1.2 or ratio <= 0.8:
            return "rectangle"
        return "unknown"


    def compute_grid_labels(self, image: np.ndarray, masks: List[np.ndarray], grid_boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
        labels = {}
        for idx, ((x0, y0), (x1, y1)) in enumerate(grid_boxes):
            best_label = "" 
            max_overlap = 0
            for mask in masks:
                cell_mask = np.zeros_like(mask, dtype=np.uint8)
                cell_mask[y0:y1, x0:x1] = 1
                intersection = np.logical_and(mask, cell_mask).sum()
                if intersection > max_overlap:
                    color, mean_hsv = self.estimate_dominant_color(image, mask & cell_mask)
                    shape = self.estimate_shape(mask)
                    best_label = f"{color} {shape}\nHSV: {int(mean_hsv[0])}, {int(mean_hsv[1])}, {int(mean_hsv[2])}"
                    max_overlap = intersection
            labels[(idx // GRID_SIZE, idx % GRID_SIZE)] = best_label
        return labels
