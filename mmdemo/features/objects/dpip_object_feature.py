import copy
import random
import threading
import time
from pathlib import Path
from typing import List, Tuple, final

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.utils.amg import MaskData, rle_to_mask

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.dpip_config import *
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    DpipActionInterface,
    DpipObjectInterface3D,
)
from mmdemo.interfaces.data import DpipGamrTarget, DpipObjectInfo3D


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
        skipPost: bool = False,
    ) -> None:
        super().__init__(color, depth, actions)
        self.all_grid_states = {}
        self.skipPost = skipPost
        self.lastCol = None
        self.xy_grid = None
        self.main_xy_grid = [["", "", ""], ["", "", ""], ["", "", ""]]
        self.xy_grid_counts = {
            (0, 0): ("", 0),
            (0, 1): ("", 0),
            (0, 2): ("", 0),
            (1, 0): ("", 0),
            (1, 1): ("", 0),
            (1, 2): ("", 0),
            (2, 0): ("", 0),
            (2, 1): ("", 0),
            (2, 2): ("", 0),
        }
        self.region_frac = DEFAULT_REGION_FRAC
        self.boxes = {}
        self.centers = {}
        self.coords = {}
        self.segmentation_masks = {}
        self.labels = {}
        self.norm_point_prompt_grid = None
        self.crop_bounds = None
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

        self.norm_point_prompt_grid = self.build_per_cell_centered_norm_point_grids(
            GRID_SIZE, POINT_PROMPTS_PER_AXIS, DEFAULT_POINT_PROMPT_GRID_REGION_FRAC
        )

        #        self.norm_point_prompt_grid = self.build_per_cell_cross_norm_point_grids(
        #            GRID_SIZE, POINT_PROMPTS_PER_AXIS, DEFAULT_POINT_PROMPT_GRID_REGION_FRAC
        #        )

        self.sam2_mask_generator = self.create_sam2_mask_generator(
            [self.norm_point_prompt_grid]
        )

    def get_output(
        self,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        actions: DpipActionInterface,
    ) -> DpipObjectInterface3D | None:
        if self.skipPost:
            return DpipObjectInterface3D(
                xyGrid=copy.deepcopy(self.main_xy_grid),
                frame_index=col.frame_count,
                region_frac=self.region_frac,
                norm_point_prompt_grid=self.norm_point_prompt_grid,
                crop_bounds=self.crop_bounds,
                labels=self.labels,
                boxes=self.boxes,
                centers=self.centers,
                coords=self.coords,
                segmentation_masks=self.segmentation_masks,
            )

        if not col.is_new() or not dep.is_new():
            return None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            self.region_frac = min(self.region_frac + REGION_FRAC_INCREMENT, 1.0)
        elif key == ord("s"):
            self.region_frac = max(self.region_frac - REGION_FRAC_INCREMENT, 0.05)

        self.lastCol = col
        if not self.t.is_alive():
            self.t = threading.Thread(target=self.worker)
            self.t.start()

        return DpipObjectInterface3D(
            xyGrid=copy.deepcopy(self.main_xy_grid),
            frame_index=col.frame_count,
            region_frac=self.region_frac,
            norm_point_prompt_grid=self.norm_point_prompt_grid,
            crop_bounds=self.crop_bounds,
            labels=self.labels,
            boxes=self.boxes,
            centers=self.centers,
            coords=self.coords,
            segmentation_masks=self.segmentation_masks,
        )

    def worker(self):
        print("\nNew Object Request Thread Started")
        try:
            self.crop_bounds = self.get_center_crop_bounds(
                self.lastCol.frame.shape, self.region_frac
            )

            all_segmentation_masks = self.get_segmentation_masks(
                self.sam2_mask_generator, self.lastCol.frame, self.crop_bounds
            )

            h, w = self.lastCol.frame.shape[:2]
            region_size = self.region_frac * min(h, w)
            cell_size = region_size / GRID_SIZE
            area_threshold = MASK_SIZE_THRESH_FRAC * cell_size**2

            filtered_masks = []
            for mask in all_segmentation_masks:
                if self.is_valid_block_mask(mask, area_threshold):
                    filtered_masks.append(mask)

            print(
                f"Kept {len(filtered_masks)} out of {len(all_segmentation_masks)} masks"
            )
            self.segmentation_masks = filtered_masks

            self.boxes, self.centers, self.coords = self.build_centered_grid_boxes(
                self.lastCol.frame.shape, GRID_SIZE, self.region_frac
            )
            self.labels = self.compute_grid_labels(
                self.lastCol.frame,
                self.segmentation_masks,
                self.boxes,
                self.region_frac,
            )

            current_xy_grid = self.grid_labels_to_xy_matrix(self.labels, GRID_SIZE)
            print(f"current xy_grid: {current_xy_grid}")

            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    current_val, current_count = self.xy_grid_counts[(x, y)]
                    if current_xy_grid[x][y] == current_val:
                        current_count += 1
                    else:
                        current_val = current_xy_grid[x][y]
                        current_count = 0
                    self.xy_grid_counts[(x, y)] = (current_val, current_count)

            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    current_val, current_count = self.xy_grid_counts[(x, y)]
                    if current_count > 3:
                        self.main_xy_grid[x][y] = current_val

            print(f"main xy_grid: {self.main_xy_grid}")

            self.xy_grid = current_xy_grid

            self.all_grid_states[self.lastCol.frame_count] = self.main_xy_grid

        except Exception as e:
            self.main_xy_grid = None
            self.region_frac = DEFAULT_REGION_FRAC
            print(f"DPIP OBJECT FEATURE THREAD: An error occurred: {e}")

    # ========== Image Processing ==========

    def apply_blur(
        self,
        image: np.ndarray,
        bilateral_iterations: int = 2,
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 20,
        bilateral_sigma_space: int = 20,
        median_ksize: int = 3,
    ):
        filtered = image.copy()
        for _ in range(bilateral_iterations):
            filtered = cv2.bilateralFilter(
                filtered, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
            )
            filtered = cv2.medianBlur(filtered, median_ksize)
        return filtered

    def get_center_crop_bounds(
        self, image_shape: Tuple[int, int], region_frac: float
    ) -> Tuple[int, int, int, int]:
        H, W = image_shape[:2]
        region_size = region_frac * min(H, W)
        cx, cy = W // 2, H // 2
        half = int(region_size / 2)
        return cx - half, cy - half, cx + half, cy + half

    # ========== Color and Shape Detection ==========

    def circular_mean_hue(self, hue_values: np.ndarray) -> float:
        hue_values = hue_values.astype(np.float32)
        angles = 2 * np.pi * hue_values / 180.0
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        mean_angle = np.arctan2(sin_sum, cos_sum)
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        return (180.0 * mean_angle) / (2 * np.pi)

    def mean_hsv_from_mask(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2HSV)
        masked_pixels = hsv[mask > 0]

        if masked_pixels.shape[0] == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        mean_h = self.circular_mean_hue(masked_pixels[:, 0])
        mean_s = masked_pixels[:, 1].mean()
        mean_v = masked_pixels[:, 2].mean()

        return np.array([mean_h, mean_s, mean_v], dtype=np.float32)

    def estimate_dominant_color(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[str, Tuple[float, float, float]]:
        mean_hsv = self.mean_hsv_from_mask(image, mask)
        mean_hue = mean_hsv[0]
        mean_saturation = mean_hsv[1]

        color_name = ""

        if mean_saturation < WHITE_BASEBOARD_SATURATION_THRESH:
            return color_name, mean_hsv

        if mean_hue < RED_MIN_HUE or mean_hue >= RED_MAX_HUE:
            color_name = "red"
        elif ORANGE_MIN_HUE <= mean_hue < ORANGE_MAX_HUE:
            color_name = "orange"
        elif YELLOW_MIN_HUE <= mean_hue < YELLOW_MAX_HUE:
            color_name = "yellow"
        elif GREEN_MIN_HUE <= mean_hue < GREEN_MAX_HUE:
            color_name = "green"
        elif BLUE_MIN_HUE <= mean_hue < BLUE_MAX_HUE:
            color_name = "blue"

        return color_name, mean_hsv

    def estimate_shape(self, mask: np.ndarray) -> str:
        shape = ""
        if self.is_mask_square(mask):
            shape = "square"
        elif self.is_mask_rectangle(mask):
            shape = "rectangle"
        return shape

    # ========== Mask Utilities ==========

    def calculate_mask_width_height_ratio(self, mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0
        biggest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        return max(w, h) / (min(w, h) + 1e-5)

    def is_mask_square(self, mask: np.ndarray) -> bool:
        width_height_ratio = self.calculate_mask_width_height_ratio(mask)
        return MIN_SQUARE_RATIO < width_height_ratio < MAX_SQUARE_RATIO

    def is_mask_rectangle(self, mask: np.ndarray) -> bool:
        width_height_ratio = self.calculate_mask_width_height_ratio(mask)
        return MIN_RECTANGLE_RATIO < width_height_ratio < MAX_RECTANGLE_RATIO

    def is_valid_block_mask(self, mask: np.ndarray, area_threshold: int):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h)

        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        solidity = area / (cv2.contourArea(cv2.convexHull(contour)) + 1e-5)

        return (
            (
                MIN_SQUARE_RATIO < aspect_ratio < MAX_SQUARE_RATIO
                or MIN_RECTANGLE_RATIO < aspect_ratio < MAX_RECTANGLE_RATIO
            )
            and area > area_threshold
            and extent > 0.9
            and solidity > 0.95
        )

    # ========== Grid Utilities ==========

    def build_centered_grid_boxes(
        self, image_shape: np.ndarray, grid_size: int, region_frac: float
    ):
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

    def compute_grid_labels(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        grid_boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        region_frac: float,
    ):
        if not masks:
            return {}

        h, w = image.shape[:2]
        region_size = region_frac * min(h, w)
        cell_size = region_size / GRID_SIZE
        cell_area_intersection_threshold = (
            CELL_AREA_INTERSECTION_THRESH_FRAC * cell_size**2
        )

        labels = {}
        for idx, ((x0, y0), (x1, y1)) in enumerate(grid_boxes):
            best_mask = None
            max_overlap = 0
            label = ""
            for mask in masks:
                cell_mask = np.zeros_like(mask, dtype=np.uint8)
                cell_mask[y0:y1, x0:x1] = 1
                intersection = np.logical_and(mask, cell_mask).sum()
                if (
                    intersection > max_overlap
                    and intersection > cell_area_intersection_threshold
                ):
                    best_mask = mask
                    max_overlap = intersection
            if best_mask is not None:
                color, mean_hsv = self.estimate_dominant_color(image, best_mask)
                shape = self.estimate_shape(best_mask)
                if shape and color:
                    label = f"{color} {shape}\nHSV: {int(mean_hsv[0])}, {int(mean_hsv[1])}, {int(mean_hsv[2])}"
            labels[(idx // GRID_SIZE, idx % GRID_SIZE)] = label
        return labels

    def grid_labels_to_xy_matrix(self, labels: dict, grid_size: int) -> list[list[str]]:
        # TODO: I feel like this function could be better. It seems redundant to try to construct the grid matrix from the labels.
        grid_matrix = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                label = labels.get((i, j), "")
                if label:
                    # Only keep the first line: "color shape"
                    label = label.split("\n")[0]
                    # Just get the first letter from the color and shape respectively
                    label = f"{label.split(' ')[0][0]}{label.split(' ')[1][0]}"
                row.append(label)
            grid_matrix.append(row)
        return grid_matrix

    def build_centered_norm_point_grid(
        self, points_per_axis: int, region_frac: float = 0.9
    ) -> np.ndarray:
        assert 0 < region_frac <= 1
        pts = np.linspace((1 - region_frac) / 2, (1 + region_frac) / 2, points_per_axis)
        xs, ys = np.meshgrid(pts, pts)
        return np.stack([xs.ravel(), ys.ravel()], axis=-1)

    def build_per_cell_centered_norm_point_grids(
        self, grid_size: int, points_per_axis: int, region_frac: float
    ) -> np.ndarray:
        assert 0 < region_frac <= 1.0

        cell_size = 1.0 / grid_size
        inner_size = region_frac * cell_size
        cell_offset = (cell_size - inner_size) / 2

        local = np.linspace(0, 1, points_per_axis)
        local_xs, local_ys = np.meshgrid(local, local)
        local_points = np.stack([local_xs.ravel(), local_ys.ravel()], axis=-1)

        all_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell_x0 = j * cell_size + cell_offset
                cell_y0 = i * cell_size + cell_offset

                for lx, ly in local_points:
                    x = cell_x0 + lx * inner_size
                    y = cell_y0 + ly * inner_size
                    all_points.append([x, y])

        return np.array(all_points, dtype=np.float32)

    def build_per_cell_cross_norm_point_grids(
        self, grid_size: int, points_per_axis: int, region_frac: float
    ) -> np.ndarray:
        assert 0 < region_frac <= 1.0

        cell_size = 1.0 / grid_size
        inner_size = region_frac * cell_size
        cell_offset = (cell_size - inner_size) / 2

        lin = np.linspace(0, 1, points_per_axis)

        horiz = np.stack([lin, np.full_like(lin, 0.5)], axis=-1)
        vert = np.stack([np.full_like(lin, 0.5), lin], axis=-1)

        local_points = np.concatenate([horiz, vert], axis=0)
        local_points = np.unique(local_points, axis=0)

        all_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell_x0 = j * cell_size + cell_offset
                cell_y0 = i * cell_size + cell_offset

                for lx, ly in local_points:
                    x = cell_x0 + lx * inner_size
                    y = cell_y0 + ly * inner_size
                    all_points.append([x, y])

        return np.array(all_points, dtype=np.float32)

    # ========== SAM2 Segmentation ==========

    def create_sam2_mask_generator(self, point_grids):
        sam2_checkpoint = (
            "C:\\Users\\Multimodal_Demo\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
        )
        sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        print(f"SAM2 device is {self.device}")

        sam2 = build_sam2(
            sam2_model_config,
            sam2_checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=None,
            point_grids=point_grids,
            pred_iou_thresh=SAM2_PREDICTED_IOU_THRESH,
            stability_score_thresh=SAM2_STABILITY_SCORE_THRESH,
            multimask_output=False,
            use_m2m=False,
        )

        return sam2_mask_generator

    def get_segmentation_masks(
        self,
        sam2_mask_generator,
        image: np.ndarray,
        crop_bounds: Tuple[int, int, int, int],
    ):
        try:
            start_time = time.time()
            H, W = image.shape[:2]

            x0, y0, x1, y1 = crop_bounds
            cropped_image = image[y0:y1, x0:x1]
            refined_image = self.apply_blur(cropped_image)

            mask_data = sam2_mask_generator._generate_masks(refined_image)

            mask_data = SAM2AutomaticMaskGenerator.postprocess_small_regions(
                mask_data, POSTPROCESS_MIN_AREA, POSTPROCESS_NMS_THRESH
            )

            segmentation_masks = [rle_to_mask(rle) for rle in mask_data["rles"]]

            aligned_masks = []
            for mask in segmentation_masks:
                full_mask = np.zeros((H, W), dtype=mask.dtype)
                full_mask[y0:y1, x0:x1] = mask
                aligned_masks.append(full_mask)

            end_time = time.time()
            print(
                f"time to compute segmentation masks: {end_time - start_time:.4f} seconds"
            )

            return aligned_masks

        except Exception as e:
            print(f"Exception while generating segmentation masks: {e}")
