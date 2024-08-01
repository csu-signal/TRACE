from pathlib import Path
from time import time
from typing import Callable

import cv2 as cv

from demo.featureModules import (AsrFeature, AsrFeatureEval, BaseDevice,
                                 CommonGroundFeature, DenseParaphrasingFeature,
                                 GazeBodyTrackingFeature, GazeFeature,
                                 GestureFeature, GestureFeatureEval, MicDevice,
                                 MoveFeature, MoveFeatureEval, ObjectFeature,
                                 ObjectFeatureEval, PoseFeature,
                                 PrerecordedDevice, PropExtractFeature,
                                 PropExtractFeatureEval)
from demo.featureModules.evaluation.eval_config import EvaluationConfig
from demo.helpers import FrameInfo, FrameTimeConverter
from demo.logger import Logger


class FeatureManager:
    def __init__(
        self,
        *,
        output_dir: Path,
        summary_log: Logger,
        error_log: Logger,
        audio_devices: list[BaseDevice],
        eval_config: EvaluationConfig | None,
    ) -> None:
        eval_dir = Path(eval_config.directory) if eval_config is not None else ""

        self.frame_time_lookup = FrameTimeConverter()

        self.summary_log = summary_log
        self.error_log = error_log

        if eval_config is not None and eval_config.objects:
            self.objects = ObjectFeatureEval(eval_dir, log_dir=output_dir)
        else:
            self.objects = ObjectFeature(log_dir=output_dir)

        shift = 7  # TODO what is this?
        self.gaze = GazeBodyTrackingFeature(shift, log_dir=output_dir)

        if eval_config is not None and eval_config.gesture:
            self.gesture = GestureFeatureEval(eval_dir, log_dir=output_dir)
        else:
            self.gesture = GestureFeature(shift, log_dir=output_dir)

        self.pose = PoseFeature(log_dir=output_dir)

        if eval_config is not None and eval_config.asr:
            self.asr = AsrFeatureEval(
                eval_dir, chunks_in_input_dir=True, log_dir=output_dir
            )
        else:
            self.asr = AsrFeature(audio_devices, n_processors=1, log_dir=output_dir)

        self.dense_paraphrasing = DenseParaphrasingFeature(log_dir=output_dir)

        if eval_config is not None and eval_config.prop:
            self.prop = PropExtractFeatureEval(eval_dir, log_dir=output_dir)
        elif eval_config is not None and eval_config.prop_model is not None:
            self.prop = PropExtractFeature(
                log_dir=output_dir, model_dir=eval_config.prop_model
            )
        else:
            self.prop = PropExtractFeature(log_dir=output_dir)

        if eval_config is not None and eval_config.move:
            self.move = MoveFeatureEval(eval_dir, log_dir=output_dir)
        elif eval_config is not None and eval_config.move_model is not None:
            self.move = MoveFeature(log_dir=output_dir, model=eval_config.move_model)
        else:
            self.move = MoveFeature(log_dir=output_dir)

        self.common_ground = CommonGroundFeature(log_dir=output_dir)

    def finalize(self):
        # TODO: join_processes=True once they are guaranteed to close cleanly
        # right now it just works for 1 processor and 1 builder
        self.asr.exit(join_processes=False)

    def processFrame(
        self, frame_info: FrameInfo, feature_active: Callable[[str], bool]
    ):
        """
        Process a frame with all the active features.
        """
        device_id = 0
        h, w, _ = frame_info.output_frame.shape

        self.frame_time_lookup.add_data(frame_info.frame_count, time())

        # run features
        blockStatus = {}
        blocks = []

        if feature_active("objects"):
            blocks = self.objects.processFrame(
                frame_info.framergb, frame_info.frame_count
            )

        if feature_active("pose"):
            self.pose.processFrame(
                frame_info.bodies,
                frame_info.output_frame,
                frame_info.frame_count,
                False,
            )

        try:
            if feature_active("gaze"):
                self.gaze.processFrame(
                    frame_info.bodies,
                    w,
                    h,
                    frame_info.rotation,
                    frame_info.translation,
                    frame_info.cameraMatrix,
                    frame_info.distortion,
                    frame_info.output_frame,
                    frame_info.framergb,
                    frame_info.depth,
                    blocks,
                    blockStatus,
                    frame_info.frame_count,
                )
        except Exception as e:
            self.error_log.append(f"Frame {frame_info.frame_count}\nGaze\n{str(e)}\n\n")

        if feature_active("gesture"):
            self.gesture.processFrame(
                device_id,
                frame_info.bodies,
                w,
                h,
                frame_info.rotation,
                frame_info.translation,
                frame_info.cameraMatrix,
                frame_info.distortion,
                frame_info.output_frame,
                frame_info.framergb,
                frame_info.depth,
                blocks,
                blockStatus,
                frame_info.frame_count,
                False,
            )

        new_utterances = []
        if feature_active("asr"):
            new_utterances = self.asr.processFrame(
                frame_info.output_frame,
                frame_info.frame_count,
                self.frame_time_lookup.get_frame,
                False,
            )

        if feature_active("dense paraphrasing"):
            self.dense_paraphrasing.processFrame(
                frame_info.output_frame,
                new_utterances,
                self.asr.utterance_lookup,
                self.gesture.blockCache,
                frame_info.frame_count,
            )

        try:
            if feature_active("prop"):
                self.prop.processFrame(
                    frame_info.output_frame,
                    new_utterances,
                    self.dense_paraphrasing.paraphrased_utterance_lookup,
                    frame_info.frame_count,
                    False,
                )
        except Exception as e:
            self.error_log.append(
                f"Frame {frame_info.frame_count}\nProp extractor\n{new_utterances}\n{str(e)}\n\n"
            )

        if feature_active("move"):
            self.move.processFrame(
                frame_info.output_frame,
                new_utterances,
                self.dense_paraphrasing.paraphrased_utterance_lookup,
                frame_info.frame_count,
                False,
            )

        if feature_active("common ground"):
            self.common_ground.processFrame(
                frame_info.output_frame,
                new_utterances,
                self.prop.prop_lookup,
                self.move.move_lookup,
                frame_info.frame_count,
            )

        cv.putText(
            frame_info.output_frame,
            "FRAME:" + str(frame_info.frame_count),
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        self._update_summary(new_utterances, frame_info.frame_count)

    def _update_summary(self, new_utterances: list[int], frame_count: int):
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
