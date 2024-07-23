import csv
from pathlib import Path

import cv2

from featureModules import PropExtractFeature
from featureModules.asr.AsrFeature import UtteranceInfo
from featureModules.prop.PropExtractFeature import PropInfo


class PropExtractFeatureEval(PropExtractFeature):
    def __init__(self, input_dir, log_dir=None):
        self.init_logger(log_dir)

        self.input_dir = Path(input_dir)

        self.prop_lookup: dict[int, PropInfo] = {}

        with open(self.input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for (frame, id, prop, text, num_props) in reader:
                prop = PropInfo(
                        int(id),
                        prop
                    )
                self.prop_lookup[prop.utterance_id] = prop

    def processFrame(self, frame, new_utterance_ids: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo], frame_count: int, includeText):

        for i in new_utterance_ids:
            utterance = utterance_lookup[i]
            self.log_prop(frame_count, self.prop_lookup[i], utterance.text, -1)

        if includeText:
            cv2.putText(frame, "Prop evaluation", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
