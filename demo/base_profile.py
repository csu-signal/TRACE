"""
Base profile which implements standard demo behavior.
"""

import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2 as cv

from demo.config import K4A_DIR, PLAYBACK_SKIP_FRAMES, PROCESSED_SIZE
from demo.feature_manager import FeatureManager
from demo.featureModules.evaluation.eval_config import EvaluationConfig
from demo.gui import Gui
from demo.helpers import FrameInfo, FrameTimeConverter, frames_to_video
from demo.logger import Logger

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(K4A_DIR)
import azure_kinect


class BaseProfile(ABC):
    """
    A base class for the demo behavior. This class creates all the necessary features
    and handles any evaluation parameters. It also handles saving frames and turning the
    output into a video.

    The two abstract methods that must be implemented by a derived class are:
    create_camera_device -- create and return an azure kinect Device
    create_audio_devices -- create and return audio devices (which inherit from BaseDevice)
    """

    def __init__(
        self,
        *,
        eval_config: EvaluationConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        if output_dir is None:
            self.output_dir = Path(
                f"stats_{str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))}"
            )
        else:
            self.output_dir = Path(output_dir)

        self.video_dir = self.output_dir / "video_files"
        self.processed_frame_dir = self.output_dir / "processed_frames"
        self.raw_frame_dir = self.output_dir / "raw_frames"

        self.eval = eval_config

        self.saved_frame_count = 0

    def run(self):
        """
        Run the full profile.
        """
        self.init_features()

        def runner():
            while self._update():
                pass

            self.finalize()

        demo_thread = Thread(target=runner)
        demo_thread.start()
        self.gui.mainloop()
        demo_thread.join()

    def _update(self):
        # skip frames to match target fps
        if isinstance(self.device, azure_kinect.Playback):
            self.device.skip_frames(PLAYBACK_SKIP_FRAMES)

        # get output from camera or playback
        color_image, depth_image, body_frame_info = self.device.get_frame()
        if color_image is None or depth_image is None:
            print(f"no color/depth image, skipping frame")
            self.fail_count += 1
            return

        self.fail_count = 0

        color_image = color_image[:, :, :3]
        frame_count = self.device.get_frame_count()

        frame_info = FrameInfo(
            output_frame=cv.cvtColor(color_image, cv.IMREAD_COLOR),
            framergb=cv.cvtColor(color_image, cv.COLOR_BGR2RGB),
            depth=depth_image,
            bodies=body_frame_info["bodies"],
            rotation=self.rotation,
            translation=self.translation,
            cameraMatrix=self.cameraMatrix,
            distortion=self.distortion,
            frame_count=frame_count,
        )

        # preprocessing
        cv.imwrite(
            f"{self.raw_frame_dir}\\frame{self.saved_frame_count:08}.png",
            frame_info.output_frame,
        )

        # process frame
        self.features.processFrame(frame_info, self.gui.feature_active)

        # postprocessing
        frame_info.output_frame = cv.resize(frame_info.output_frame, PROCESSED_SIZE)
        self.gui.new_image(cv.cvtColor(frame_info.output_frame, cv.COLOR_BGR2RGB))
        cv.imwrite(
            f"{self.processed_frame_dir}\\frame{self.saved_frame_count:08}.png",
            frame_info.output_frame,
        )
        self.saved_frame_count += 1

        # check if done
        if self.is_done(frame_count, self.fail_count):
            return False

        return True

    def init_features(self):
        """
        Initialize features. This method should be used as opposed to __init__ to stop
        unused profiles from initializing all features and taking up memory.
        """
        for i in (
            self.output_dir,
            self.video_dir,
            self.processed_frame_dir,
            self.raw_frame_dir,
        ):
            os.makedirs(i, exist_ok=True)

        self.error_log = Logger(file=self.output_dir / "errors.txt", stdout=True)
        self.summary_log = Logger(file=self.output_dir / "summary.txt", stdout=True)
        self.error_log.clear()
        self.summary_log.clear()

        self.features = FeatureManager(
            output_dir=self.output_dir,
            summary_log=self.summary_log,
            error_log=self.error_log,
            audio_devices=self.create_audio_devices(),
            eval_config=self.eval,
        )

        # get device information
        self.device = self.create_camera_device()
        self.cameraMatrix, self.rotation, self.translation, self.distortion = (
            self.device.get_calibration_matrices()
        )
        self.fail_count = 0

        self.gui = Gui()

    def finalize(self):
        """
        Run after the processing is complete. Turns the saved frames into videos and removes
        the frame directories to clear up space.
        """
        self.device.close()
        self.features.finalize()
        if self.gui.running:
            self.gui.close()

        frames_to_video(
            f"{self.processed_frame_dir}\\frame%8d.png",
            f"{self.video_dir}\\processed_frames.mp4",
        )
        frames_to_video(
            f"{self.raw_frame_dir}\\frame%8d.png", f"{self.video_dir}\\raw_frames.mp4"
        )

        # remove frame images (they take a lot of space)
        shutil.rmtree(self.processed_frame_dir)
        shutil.rmtree(self.raw_frame_dir)

    def is_done(self, frame_count, fail_count):
        """
        Return a boolean which is true if the demo should stop processing. This happens
        when either the display window is manually closed or 20 frames have failed in a row.

        Arguments:
        frame_count -- the current frame
        fail_count -- the number of consecutive failed frames
        """
        return not self.gui.running or fail_count > 20

    @abstractmethod
    def create_camera_device(self) -> azure_kinect.Device:
        raise NotImplementedError

    @abstractmethod
    def create_audio_devices(self):
        raise NotImplementedError
