from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    Vectors3DInterface,
)
from mmdemo.utils.cone_shape import ConeShape
from mmdemo.utils.support_utils import Joint
from mmdemo.utils.threeD_object_loc import checkBlocks
from mmdemo.utils.twoD_object_loc import convert2D
