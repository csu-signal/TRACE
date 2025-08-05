from typing import final

import cv2

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface


@final
class WebcamDevice(BaseFeature[ColorImageInterface]):
    def __init__(self, camera_index: int | None):
        super().__init__()
        self.camera_index = camera_index
        self.frame_count = 0
        self.webcam_stopped = False

    def initialize(self):
        assert self.camera_index is not None, "Camera index is required"
        self.cap = cv2.VideoCapture(self.camera_index)
        assert self.cap.isOpened(), "Webcam not accessible"

    def finalize(self):
        self.cap.release()

    def get_output(self):
        ret, frame = self.cap.read()
        if not ret:
            self.webcam_stopped = True
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        color = ColorImageInterface(self.frame_count, rgb_frame)
        self.frame_count += 1
        return color

    def is_done(self) -> bool:
        return self.webcam_stopped
