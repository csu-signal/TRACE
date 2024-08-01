"""
Helper classes and functions for the demo
"""

import os
from dataclasses import dataclass

import numpy as np
from config import PLAYBACK_TARGET_FPS
from cv2.typing import MatLike


@dataclass
class FrameInfo:
    output_frame: MatLike
    framergb: MatLike
    depth: MatLike
    bodies: list
    rotation: np.ndarray
    translation: np.ndarray
    cameraMatrix: np.ndarray
    distortion: np.ndarray
    frame_count: int


class FrameTimeConverter:
    """
    A helper class that can quickly look up what time a frame was processed
    or which frame was being processed at a given time.
    """

    def __init__(self) -> None:
        self.data = []

    def add_data(self, frame, time):
        """
        Add a new datapoint. The frame and time must be strictly increasing
        so binary search can be used.

        Arguments:
        frame -- the frame number
        time -- the current time
        """
        # must be strictly monotonic so binary search can be used
        assert len(self.data) == 0 or frame > self.data[-1][0]
        assert len(self.data) == 0 or time > self.data[-1][1]
        self.data.append((frame, time))

    def get_time(self, frame):
        """
        Return the time that a frame was processed
        """
        return self._binary_search(0, frame)[1]

    def get_frame(self, time):
        """
        Return the frame being processed at a certain time
        """
        return self._binary_search(1, time)[0]

    def _binary_search(self, index, val):
        assert len(self.data) > 0
        assert self.data[-1][index] >= val
        left = 0
        right = len(self.data)
        while right - left > 1:
            middle = (left + right) // 2
            if self.data[middle][index] < val:
                left = middle
            elif self.data[middle][index] > val:
                right = middle
            else:
                left = middle
                right = middle
        return self.data[left]


def frames_to_video(frame_path, output_path, rate=PLAYBACK_TARGET_FPS):
    """
    Given a path representing a series of images, create a video.

    Arguments:
    frame_path -- a format string representing the image paths (ex. "frame%8d.png")
    output_path -- the path of the output video (ex. "result.mp4")
    rate -- the frame rate of the output video, defaults to PLAYBACK_TARGET_FPS
    """
    os.system(
        f"ffmpeg -framerate {rate} -i {frame_path} -c:v libx264 -pix_fmt yuv420p {output_path}"
    )
