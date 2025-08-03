from pathlib import Path
import threading
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
    DpipActionInterface,
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
DEFAULT_REGION_FRAC = 0.35 # changing this to 0.35 since a larger grid box is going beyond the base board - Sifat
COLOR_THRESHOLDS = {
    "red": (114, 127),
    "orange": (105, 114),
    "yellow": (93, 105),
    "green": (25, 55),
    "blue": (6, 15),
}

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
        actions: BaseFeature[DpipActionInterface],
        *,
        detection_threshold=0.6,
        model_path: Path | None = None,
        skipPost: bool=False
    ) -> None:
        super().__init__(color, depth, actions)
        self.all_grid_states = {}
        self.skipPost = skipPost
        self.lastCol = None
        self.xy_grid = None
        self.region_frac = DEFAULT_REGION_FRAC
        self.boxes = {}
        self.centers = {}
        self.coords = {}
        self.segmentation_masks = {}
        self.labels = {}
        self.t = threading.Thread(target=self.worker)
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
        actions: DpipActionInterface,
    ) -> DpipObjectInterface3D | None:
        if(self.skipPost):
            return DpipObjectInterface3D(xyGrid=self.xy_grid, frame_index=col.frame_count, region_frac=self.region_frac, labels=self.labels, boxes=self.boxes, centers=self.centers, coords=self.coords, segmentation_masks = self.segmentation_masks)
    
        if not col.is_new() or not dep.is_new():
            return None
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            self.region_frac = min(self.region_frac + 0.05, 1.0)
        elif key == ord('s'):
            self.region_frac = max(self.region_frac - 0.05, 0.05)
       
        self.lastCol = col 
        if not self.t.is_alive():
            self.t = threading.Thread(target=self.worker)
            self.t.start()

        return DpipObjectInterface3D(xyGrid=self.xy_grid, frame_index=col.frame_count, region_frac=self.region_frac, labels=self.labels, boxes=self.boxes, centers=self.centers, coords=self.coords, segmentation_masks = self.segmentation_masks)

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
            # cy, cx = H * 0.3, W // 2
            half = int(region_size / 2)
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            cropped_image = image[y0:y1, x0:x1]

            blurred_cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 1)
            # Bilateral filter supposedly preserves edges
            #blurred_cropped_image = cv2.bilateralFilter(cropped_image, 9, 75, 75)

            sam2_output = sam2_mask_generator.generate(blurred_cropped_image)
            segmentation_masks = [mask["segmentation"] for mask in sam2_output]

            # To avoid returning maps of objects smaller than 50% of one grid square - Sifat
            cropped_area_of_image = (y1 - y0) * (x1 - x0)
            min_mask_area = 0.5 * (cropped_area_of_image / (GRID_SIZE ** 2))

            aligned_masks = []
            for mask in segmentation_masks:
                mask_area = np.sum(mask)
                if mask_area < min_mask_area:
                    # print("map too small") # for debugging, please comment out - Sifat
                    continue
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
                    label = "none square"
                # Just get the first letter from the color and shape respectively
                if len(label.split(" ")) == 2:
                    row.append(f"{label.split(' ')[0][0]}{label.split(' ')[1][0]}")
            grid_matrix.append(row)
        return grid_matrix


    def build_centered_grid_boxes(self, image_shape, grid_size, region_frac):
        h, w = image_shape[:2]
        region_size = region_frac * min(h, w)
        cell_size = region_size / grid_size
        cy, cx = h / 2, w / 2
        # cy, cx = (h * 0.3), w / 2
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

        # if mean_hue < 10 or mean_hue >= 160:
        #     color_name = "red"
        # elif 10 <= mean_hue < 20:
        #     color_name = "orange"
        # elif 20 <= mean_hue < 35:
        #     color_name = "yellow"
        # elif 35 <= mean_hue < 85:
        #     color_name = "green"
        # elif 85 <= mean_hue < 170:
        #     color_name = "blue"
        # else:
        #     color_name = "unknown"

        color_name = "unknown"
        for name, (low, high) in COLOR_THRESHOLDS.items():
            if low <= mean_hue < high:
                color_name = name
                break
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
                grid_area = (y1 - y0) * (x1 - x0)
                # if mask doesn't cover 50% of square, ignore - Sifat
                if intersection < 0.5 * grid_area:
                    continue 
                if intersection > max_overlap:
                    color, mean_hsv = self.estimate_dominant_color(image, mask & cell_mask)
                    shape = self.estimate_shape(mask)
                    # best_label = f"{color} {shape}\nHSV: {int(mean_hsv[0])}, {int(mean_hsv[1])}, {int(mean_hsv[2])}"
                    best_label = f"{color} {shape}\nHue: {int(mean_hsv[0])}" # Only printing hue since we don't do anything with sat and val - Sifat
                    max_overlap = intersection

            # for debugging raw coverage constraint - Sifat
            if max_overlap > 0:
                coverage_percent = (max_overlap / grid_area) * 100
                print(f"Grid cell ({idx // GRID_SIZE}, {idx % GRID_SIZE}) coverage: {coverage_percent:.2f}%")

            labels[(idx // GRID_SIZE, idx % GRID_SIZE)] = best_label
        return labels
    
    def worker(self):
        print("\nNew Object Request Thread Started")
        try:
            h, w, _ = self.lastCol.frame.shape
            sam2_mask_generator = self.create_sam2_mask_generator()

            self.segmentation_masks = self.get_segmentation_masks(sam2_mask_generator, self.lastCol.frame, self.region_frac)

            self.boxes, self.centers, self.coords = self.build_centered_grid_boxes(self.lastCol.frame.shape, GRID_SIZE, self.region_frac)
            self.labels = self.compute_grid_labels(self.lastCol.frame, self.segmentation_masks, self.boxes)

            self.xy_grid = self.grid_labels_to_xy_matrix(self.labels, GRID_SIZE)
            print(self.xy_grid)
            self.all_grid_states[self.lastCol.frame_count] = self.xy_grid
            
        except Exception as e:
            self.xy_grid = None
            self.region_frac = DEFAULT_REGION_FRAC
            print(f"DPIP OBJECT FEATURE THREAD: An error occurred: {e}")
