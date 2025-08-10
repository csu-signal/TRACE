from typing import final

import cv2
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, DepthImageInterface


@final
class MetricDepthVisualization(BaseFeature[ColorImageInterface]):
    """
    Get a visualization of a metric depth map from Depth Anything v2.

    Input interface is DepthImageInterface.

    Output interface is ColorImageInterface.
    """

    def __init__(
        self,
        depth: BaseFeature[DepthImageInterface],
    ) -> None:
        super().__init__(depth)

    def get_output(
        self,
        depth: DepthImageInterface,
    ) -> ColorImageInterface | None:
        if not depth.is_new():
            return None

        normalized_depth = cv2.normalize(depth.frame, None, 0, 255, cv2.NORM_MINMAX)
        inverted_normalized_depth = 255 - normalized_depth
        depth_visualization = inverted_normalized_depth.astype(np.uint8)
        bgr_depth_visualization = cv2.applyColorMap(
            depth_visualization, cv2.COLORMAP_INFERNO
        )
        rgb_depth_visualization = cv2.cvtColor(
            bgr_depth_visualization, cv2.COLOR_BGR2RGB
        )

        return ColorImageInterface(
            frame_count=depth.frame_count, frame=rgb_depth_visualization
        )


@final
class RelativeDepthVisualization(BaseFeature[ColorImageInterface]):
    """
    Get a visualization of a relative depth map from Depth Anything v2.

    Input interface is DepthImageInterface.

    Output interface is ColorImageInterface.
    """

    def __init__(
        self,
        depth: BaseFeature[DepthImageInterface],
    ) -> None:
        super().__init__(depth)

    def get_output(
        self,
        depth: DepthImageInterface,
    ) -> ColorImageInterface | None:
        if not depth.is_new():
            return None

        normalized_depth = cv2.normalize(depth.frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_visualization = normalized_depth.astype(np.uint8)
        bgr_depth_visualization = cv2.applyColorMap(
            depth_visualization, cv2.COLORMAP_INFERNO
        )
        rgb_depth_visualization = cv2.cvtColor(
            bgr_depth_visualization, cv2.COLOR_BGR2RGB
        )

        return ColorImageInterface(
            frame_count=depth.frame_count, frame=rgb_depth_visualization
        )
