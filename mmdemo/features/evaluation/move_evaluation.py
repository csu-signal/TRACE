import csv

import cv2
from demo.featureModules import MoveFeature
from demo.featureModules.asr.AsrFeature import UtteranceInfo
from demo.featureModules.move.MoveFeature import MoveInfo


class MoveFeatureEval(MoveFeature):
    LOG_FILE = "moveOutput.csv"

    def __init__(self, input_dir, log_dir=None):
        self.init_logger(log_dir)

        self.move_lookup: dict[int, MoveInfo] = {}

        with open(input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i: j for i, j in zip(keys, row)}
                utterance_id = int(data["utterance_id"])
                include_statement = data["statement"] == "1"
                include_accept = data["accept"] == "1"
                include_doubt = data["doubt"] == "1"

                move = []
                if include_statement:
                    move.append("STATEMENT")
                if include_accept:
                    move.append("ACCEPT")
                if include_doubt:
                    move.append("DOUBT")

                self.move_lookup[utterance_id] = MoveInfo(utterance_id, move)

    def processFrame(
        self,
        frame,
        new_utterances: list[int],
        utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo],
        frameIndex,
        includeText,
    ):
        for i in new_utterances:
            text = utterance_lookup[i].text
            self.log_move(frameIndex, self.move_lookup[i], -1, text, -1)

        if includeText:
            cv2.putText(
                frame,
                "Move evaluation",
                (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
