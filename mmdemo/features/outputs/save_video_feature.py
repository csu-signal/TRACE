import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import final

import cv2 as cv

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, EmptyInterface
from mmdemo.utils.files import create_tmp_dir


@final
class SaveVideo(BaseFeature[EmptyInterface]):
    """
    Save a video by accumulating color frames.

    Input interface is `ColorImageInterface`

    Output interface is `EmptyInterface`

    Keyword arguments:
    frame_rate -- the frame rate of the input frames (default 30)
    video_name -- if not None, the name of the output video (default None)
    """

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        *,
        frame_rate=30,
        video_name: Path | int | None = None,
    ):
        super().__init__(color)

        self.frame_rate = frame_rate

        if video_name is None:
            self.video_name = Path(
                "output-video-"
                + datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S-" + ".mp4")
            )
        elif video_name == 2:
            self.video_name = Path(
                "output-video-"
                + datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S-" + "SecondaryCamera" + ".mp4")
            )
        else:
            self.video_name = Path(video_name)

    def initialize(self):
        self.tmp_dir = create_tmp_dir()
        self.counter = 0

    def finalize(self):
        frame_path = self.tmp_dir / "frame%8d.png"
        os.system(
            f"ffmpeg -framerate {self.frame_rate} -i {frame_path} -c:v libx264 -pix_fmt yuv420p {self.video_name}"
        )
        shutil.rmtree(self.tmp_dir)

    def get_output(
        self,
        color: ColorImageInterface,
    ):
        if not color.is_new():
            return None

        im = cv.cvtColor(color.frame, cv.COLOR_RGB2BGR)
        cv.imwrite(str(self.tmp_dir / f"frame{self.counter:08}.png"), im)

        self.counter += 1

        return EmptyInterface()
