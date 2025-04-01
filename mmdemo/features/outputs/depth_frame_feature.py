from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    DepthImageInterface
)

@final
class DepthFrame(BaseFeature[DepthImageInterface]):
    """
    Return the depth output

    Input interfaces are `DepthImageInterface`

    Output interface is `DepthImageInterface`
    """

    def __init__(
        self,
        depth: BaseFeature[DepthImageInterface]
    ):
        super().__init__(depth)

    def initialize(self):
        self.has_cgt_data = False
        
    def get_output(
        self,
        depth: DepthImageInterface
    ):
        if (
            not depth.is_new()
        ):
            return None

        # ensure we are not modifying the color frame itself
        output_frame = np.copy(depth.frame)
        depth_image_8bit = cv.convertScaleAbs(output_frame, alpha=(255.0/65535.0))

        depth_image_colorized = cv.applyColorMap(depth_image_8bit, cv.COLORMAP_JET)
        depth_image_colorized = cv.resize(depth_image_colorized, (1280, 720))

        return DepthImageInterface(frame=depth_image_colorized, frame_count=depth.frame_count)