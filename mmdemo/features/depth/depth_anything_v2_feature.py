import time
from typing import final

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, DepthImageInterface


class DepthAnythingV2Base(BaseFeature[DepthImageInterface]):
    """
    Base class for getting a depth map using Depth Anything v2.

    Input interface is ColorImageInterface.

    Output interface is DepthImageInterface.
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        *,
        skipPost: bool = False,
    ) -> None:
        super().__init__(color)
        self.depth_map = None
        self.skipPost = skipPost
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_output(
        self,
        col: ColorImageInterface,
    ) -> DepthImageInterface | None:
        if self.skipPost:
            return DepthImageInterface(
                frame_count=col.frame_count, frame=self.depth_map
            )

        if not col.is_new():
            return None

        self.depth_map = self.get_depth_map(col.frame)

        return DepthImageInterface(frame_count=col.frame_count, frame=self.depth_map)

    @torch.no_grad()
    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        start_time = time.time()
        inputs = self.depth_image_processor(images=image, return_tensors="pt").to(
            self.device
        )
        original_size = image.shape[:2]

        outputs = self.depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        end_time = time.time()
        print(f"time to compute depth map: {end_time - start_time:.4f} seconds")

        return depth_resized.squeeze().cpu().numpy()


@final
class DepthAnythingV2Metric(DepthAnythingV2Base):
    """
    Gets a metric depth map using Depth Anything v2.

    Input interface is ColorImageInterface.

    Output interface is DepthImageInterface.
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
    ) -> None:
        super().__init__(color)

    def initialize(self):
        self.depth_image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        ).to(self.device)


@final
class DepthAnythingV2Relative(DepthAnythingV2Base):
    """
    Gets a relative depth map using Depth Anything v2.

    Input interface is ColorImageInterface.

    Output interface is DepthImageInterface.
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
    ) -> None:
        super().__init__(color)

    def initialize(self):
        self.depth_image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        ).to(self.device)
