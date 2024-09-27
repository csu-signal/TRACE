from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface


@final
class FakeFeature(BaseFeature):
    """
    Fake feature that can be used for testing.

    Some features require other features to initialize,
    so this can be used in place of a real feature.
    """

    def get_output(self, *args):
        return None


class ColorFrameCount(BaseFeature[ColorImageInterface]):
    """
    Feature to control the frame count. There is no input
    interface and the output interface is ColorImageInterface.

    Keyword arguments:
    `frames` -- a list of frame counts. This feature will output
    these counts one after another.
    """

    def __init__(self, *, frames: list[int]) -> None:
        super().__init__()
        self.frames = frames

    def initialize(self):
        self.it = iter(self.frames)
        self.next = next(self.it)
        self.done = False

    def get_output(self):
        output = self.next

        try:
            self.next = next(self.it)
        except StopIteration:
            self.done = True

        return ColorImageInterface(frame=np.zeros((5, 5, 3)), frame_count=output)

    def is_done(self):
        return self.done
