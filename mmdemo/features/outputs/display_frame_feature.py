import random
from typing import final

import cv2 as cv

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, EmptyInterface


@final
class DisplayFrame(BaseFeature[EmptyInterface]):
    """
    Show a color frame with opencv. The demo will exit once
    the window is closed.

    Input interface is `ColorImageInterface`

    Output interface is `EmptyInterface`
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
    ):
        super().__init__(color)

    def initialize(self):
        self.window_name = str(random.random())

    def get_output(
        self,
        color: ColorImageInterface,
    ):
        if not color.is_new():
            return None

        bgr = cv.cvtColor(color.frame, cv.COLOR_RGB2BGR)
        cv.imshow(self.window_name, bgr)
        cv.waitKey(1)

        return EmptyInterface()

    def is_done(self):
        return cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 1
