import csv
from pathlib import Path

import cv2
from demo.featureModules import PropExtractFeature
from demo.featureModules.asr.AsrFeature import UtteranceInfo
from demo.featureModules.prop.PropExtractFeature import PropInfo


class PropExtractFeatureEval(PropExtractFeature):
    def __init__(self, input_dir, log_dir=None):
        self.init_logger(log_dir)

        self.input_dir = Path(input_dir)

        self.prop_lookup: dict[int, PropInfo] = {}

        with open(self.input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i: j for i, j in zip(keys, row)}
                prop = PropInfo(int(data["utterance_id"]), data["proposition"])
                self.prop_lookup[prop.utterance_id] = prop

    def processFrame(
        self,
        frame,
        new_utterance_ids: list[int],
        utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo],
        frame_count: int,
        includeText,
    ):
        for i in new_utterance_ids:
            utterance = utterance_lookup[i]
            self.log_prop(frame_count, self.prop_lookup[i], utterance.text, -1)

        if includeText:
            cv2.putText(
                frame,
                "Prop evaluation",
                (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
