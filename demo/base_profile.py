"""
Base profile which implements standard demo behavior.
"""

from dataclasses import dataclass
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path ###Path is a concrete path which include I/O
from time import time
from tkinter import Checkbutton, IntVar, Tk
import shutil
import numpy as np
import csv ###added

import cv2 as cv
from cv2.typing import MatLike

from demo.config import K4A_DIR, PLAYBACK_SKIP_FRAMES, PLAYBACK_TARGET_FPS
from demo.featureModules import (AsrFeature, AsrFeatureEval, BaseDevice,
                            CommonGroundFeature, DenseParaphrasingFeature,
                            GazeBodyTrackingFeature, GazeFeature,
                            GestureFeature, GestureFeatureEval, MicDevice,
                            MoveFeature, MoveFeatureEval, ObjectFeature,
                            ObjectFeatureEval, PoseFeature, PrerecordedDevice,
                            PropExtractFeature, PropExtractFeatureEval,
                            rec_common_ground)
from demo.featureModules.evaluation.eval_config import EvaluationConfig
from demo.logger import Logger

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(K4A_DIR)
import azure_kinect

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
            middle = (left + right)//2
            if self.data[middle][index] < val:
                left = middle
            elif self.data[middle][index] > val:
                right = middle
            else:
                left = middle
                right = middle
        return self.data[left]

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
            eval_config: EvaluationConfig|None = None,
            output_dir: str|Path|None = None
        ) -> None:
        if output_dir is None:
            self.output_dir = Path(f"stats_{str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))}")
        else:
            self.output_dir = Path(output_dir)

        self.video_dir = self.output_dir / "video_files"
        self.processed_frame_dir = self.output_dir / "processed_frames"
        self.raw_frame_dir = self.output_dir / "raw_frames"

        self.eval = eval_config
        self.eval_dir = Path(self.eval.directory) if self.eval is not None else ""

        self.frame_time_lookup = FrameTimeConverter()

        self.saved_frame_count = 0

    def init_features(self):
        """
        Initialize features. This method should be used as opposed to __init__ to stop
        unused profiles from initializing all features and taking up memory.
        """
        for i in (self.output_dir, self.video_dir, self.processed_frame_dir, self.raw_frame_dir):
            os.makedirs(i, exist_ok=True)

        self.root = Tk()
        # self.root.geometry('350x200')
        self.root.title("Output Options") ###title of the Tk() GUI window
        '''###set all but gaze to zero so that the inputs needed for the other features did not need to be created
        self.vars = {
                "gesture": IntVar(value=1),
                "objects": IntVar(value=1),
                "gaze": IntVar(value=1),
                "asr": IntVar(value=1),
                "dense paraphrasing": IntVar(value=1),
                "pose": IntVar(value=0),
                "prop": IntVar(value=1),
                "move": IntVar(value=1),
                "common ground": IntVar(value=1),
                }
        '''
        self.vars = {
                "gesture": IntVar(value=0),
                "objects": IntVar(value=0),
                "gaze": IntVar(value=1),
                "asr": IntVar(value=0),
                "dense paraphrasing": IntVar(value=0),
                "pose": IntVar(value=0),
                "prop": IntVar(value=0),
                "move": IntVar(value=0),
                "common ground": IntVar(value=0),
                }
        self._create_buttons()
        '''
        if self.eval is not None and self.eval.objects:
            self.objects = ObjectFeatureEval(self.eval_dir, log_dir=self.output_dir)
        else:
            self.objects = ObjectFeature(log_dir=self.output_dir)
        '''
        shift = 7 # TODO what is this?
        self.gaze = GazeBodyTrackingFeature(shift, log_dir=self.output_dir)
        '''
        if self.eval is not None and self.eval.gesture:
            self.gesture = GestureFeatureEval(self.eval_dir, log_dir=self.output_dir)
        else:
            self.gesture = GestureFeature(shift, log_dir=self.output_dir)

        self.pose = PoseFeature(log_dir=self.output_dir)

        if self.eval is not None and self.eval.asr:
            self.asr = AsrFeatureEval(self.eval_dir, chunks_in_input_dir=True, log_dir=self.output_dir)
        else:
            self.asr = AsrFeature(self.create_audio_devices(), n_processors=1, log_dir=self.output_dir)

        self.dense_paraphrasing = DenseParaphrasingFeature(log_dir=self.output_dir)

        if self.eval is not None and self.eval.prop:
            self.prop = PropExtractFeatureEval(self.eval_dir, log_dir=self.output_dir)
        elif self.eval is not None and self.eval.prop_model is not None:
            self.prop = PropExtractFeature(log_dir=self.output_dir, model_dir=self.eval.prop_model)
        else:
            self.prop = PropExtractFeature(log_dir=self.output_dir)

        if self.eval is not None and self.eval.move:
            self.move = MoveFeatureEval(self.eval_dir, log_dir=self.output_dir)
        elif self.eval is not None and self.eval.move_model is not None:
            self.move = MoveFeature(log_dir=self.output_dir, model=self.eval.move_model)
        else:
            self.move = MoveFeature(log_dir=self.output_dir)

        self.common_ground = CommonGroundFeature(log_dir=self.output_dir)

        if self.output_dir is not None:
            self.error_log = Logger(file=self.output_dir / "errors.txt", stdout=True)
            self.summary_log = Logger(file=self.output_dir / "summary.txt", stdout=True)
        else:
            self.error_log = Logger(stdout=True)
            self.summary_log = Logger(stdout=True)

        self.error_log.clear()
        self.summary_log.clear()
        '''
    def _create_buttons(self):
        for text,var in self.vars.items():
            Checkbutton(self.root, text=text, variable=var, onvalue=1, offvalue=0, height=2, width=10).pack()

    def _should_process(self, var):
        return self.vars[var].get()

    def processFrame(self, frame_info: FrameInfo):
        """
        Process a frame with all the active features.
        """
        device_id = 0
        h,w,_ = frame_info.output_frame.shape
        print("output_frame.shape",frame_info.output_frame.shape) ###debug
        '''
        if frame_info.frame_count < 10: ###debug
            with open('output_frame.csv','w') as f:
                writer = csv.writer(f)
                writer.writerows(frame_info.output_frame[:10][:])
                print("len dimensions",len(frame_info.output_frame),len(frame_info.output_frame[0]),len(frame_info.output_frame[0][0]))
                print("h,w",h,w)
        '''
        ###cd print(frame_info.frame_count,end=" ")

        self.root.update() ###updates and processes all the events or tasks such as redrawing widgets, geometry management, configuring the widget property

        self.frame_time_lookup.add_data(frame_info.frame_count, time())

        # run features
        blockStatus = {}
        blocks = []
        '''
        ###debug prints
        print("base_profile: objects",self._should_process("objects"))
        print("base_profile: pose",self._should_process("pose"))
        print("base_profile: gaze",self._should_process("gaze"))
        print("base_profile: gesture",self._should_process("gesture"))
        print("base_profile: asr",self._should_process("asr"))
        print("base_profile: dense paraphrasing",self._should_process("dense paraphrasing"))
        print("base_profile: prop",self._should_process("prop"))
        print("base_profile: move",self._should_process("move"))
        print("base_profile: common ground",self._should_process("common ground"))
        '''
        if(self._should_process("objects")):
            blocks = self.objects.processFrame(frame_info.framergb, frame_info.frame_count)

        if(self._should_process("pose")):
            self.pose.processFrame(frame_info.bodies, frame_info.output_frame, frame_info.frame_count, False)

        try:
            if(self._should_process("gaze")):
                self.gaze.processFrame( frame_info.bodies, w, h, frame_info.rotation, frame_info.translation, frame_info.cameraMatrix, frame_info.distortion, frame_info.output_frame, frame_info.framergb, frame_info.depth, blocks, blockStatus, frame_info.frame_count)
                ###print("isinstance np.ndarray",isinstance(frame_info.depth, np.ndarray), frame_info.depth.shape)
                ###print("frame_info.depth\n",frame_info.depth[:][:10])
                ###with open('depth_info.csv','w') as f:
                ###    writer = csv.writer(f)
                ###    writer.writerows(frame_info.depth[:][:10])
        except:
            pass
        
        if(self._should_process("gesture")):
            self.gesture.processFrame(device_id, frame_info.bodies, w, h, frame_info.rotation, frame_info.translation, frame_info.cameraMatrix, frame_info.distortion, frame_info.output_frame, frame_info.framergb, frame_info.depth, blocks, blockStatus, frame_info.frame_count, False)

        new_utterances = []
        if(self._should_process("asr")):
            new_utterances = self.asr.processFrame(frame_info.output_frame, frame_info.frame_count, self.frame_time_lookup.get_frame, False)

        if self._should_process("dense paraphrasing"):
            self.dense_paraphrasing.processFrame(frame_info.output_frame, new_utterances, self.asr.utterance_lookup, self.gesture.blockCache, frame_info.frame_count)

        try:
            if(self._should_process("prop")):
                self.prop.processFrame(frame_info.output_frame, new_utterances, self.dense_paraphrasing.paraphrased_utterance_lookup, frame_info.frame_count, False)
        except Exception as e:
            self.error_log.append(f"Frame {frame_info.frame_count}\nProp extractor\n{new_utterances}\n{str(e)}\n\n")

        if(self._should_process("move")):
            self.move.processFrame(frame_info.output_frame, new_utterances, self.dense_paraphrasing.paraphrased_utterance_lookup, frame_info.frame_count, False)

        if self._should_process("common ground"):
            self.common_ground.processFrame(frame_info.output_frame, new_utterances, self.prop.prop_lookup, self.move.move_lookup, frame_info.frame_count)


        cv.putText(frame_info.output_frame, "FRAME:" + str(frame_info.frame_count), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

        self.update_summary(new_utterances, frame_info.frame_count)

    def update_summary(self, new_utterances: list[int], frame_count: int):
        """
        Log a message after each frame.

        Arguments:
        new_utterances -- a list of utterance ids
        frame_count -- the current frame
        """
        for i in new_utterances:
            utterance = self.dense_paraphrasing.paraphrased_utterance_lookup[i]
            prop = self.prop.prop_lookup[i]
            move = self.move.move_lookup[i]

            update = ""
            update += "FRAME: " + str(frame_count) + "\n"
            update += "E bank\n"
            update += str(self.common_ground.closure_rules.ebank) + "\n"
            update += "F bank\n"
            update += str(self.common_ground.closure_rules.fbank) + "\n"
            if prop.prop == "no prop":
                update += f"{utterance.speaker_id}: {utterance.text} ({self.common_ground.most_recent_prop}), {move.move}\n\n"
            else:
                update += f"{utterance.speaker_id}: {utterance.text} => {prop.prop}, {move.move}\n\n"

            self.summary_log.append(update)

    def finalize(self):
        """
        Run after the processing is complete. Turns the saved frames into videos and removes
        the frame directories to clear up space.
        """
        # TODO: join_processes=True once they are guaranteed to close cleanly
        # right now it just works for 1 processor and 1 builder
        self.asr.exit(join_processes=False)
        self.root.destroy()

        self.frames_to_video(
                f"{self.processed_frame_dir}\\frame%8d.png",
                f"{self.video_dir}\\processed_frames.mp4")
        self.frames_to_video(
                f"{self.raw_frame_dir}\\frame%8d.png",
                f"{self.video_dir}\\raw_frames.mp4")

        # remove frame images (they take a lot of space)
        shutil.rmtree(self.processed_frame_dir)
        shutil.rmtree(self.raw_frame_dir)

    def preprocess(self, frame_info: FrameInfo):
        """Run before processFrame. Save the original image."""
        cv.imwrite(f"{self.raw_frame_dir}\\frame{self.saved_frame_count:08}.png", frame_info.output_frame)

    def postprocess(self, frame_info: FrameInfo):
        """Run after processFrame. Save and display the output image."""
        # resize, display, and save output
        frame_info.output_frame = cv.resize(frame_info.output_frame, (1280, 720))
        ###cv.imshow("output", frame_info.output_frame)
        ###cv.waitKey(1)
        cv.imwrite(f"{self.processed_frame_dir}\\frame{self.saved_frame_count:08}.png", frame_info.output_frame)

        self.saved_frame_count += 1

    def is_done(self, frame_count, fail_count):
        """
        Return a boolean which is true if the demo should stop processing. This happens
        when either the display window is manually closed or 20 frames have failed in a row.

        Arguments:
        frame_count -- the current frame
        fail_count -- the number of consecutive failed frames
        """
        print("base_profile: cv.WND_PROP_VISIBLE", cv.WND_PROP_VISIBLE)
        print("display window manually closed",cv.getWindowProperty("output", cv.WND_PROP_VISIBLE))
        return cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) == 0 or fail_count > 20

    @staticmethod
    def frames_to_video(frame_path, output_path, rate=PLAYBACK_TARGET_FPS):
        """
        Given a path representing a series of images, create a video.

        Arguments:
        frame_path -- a format string representing the image paths (ex. "frame%8d.png")
        output_path -- the path of the output video (ex. "result.mp4")
        rate -- the frame rate of the output video, defaults to PLAYBACK_TARGET_FPS
        """
        os.system(f"ffmpeg -framerate {rate} -i {frame_path} -c:v libx264 -pix_fmt yuv420p {output_path}")

@abstractmethod
def create_camera_device(self) -> azure_kinect.Device:
    raise NotImplementedError

    @abstractmethod
    def create_audio_devices(self):
        raise NotImplementedError
