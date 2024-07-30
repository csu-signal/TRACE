import csv
from collections import defaultdict
from pathlib import Path

import cv2

from demo.featureModules import AsrFeature
from demo.featureModules.asr.AsrFeature import UtteranceInfo


class AsrFeatureEval(AsrFeature):
    def __init__(self, input_dir, chunks_in_input_dir=False, log_dir=None):
        self.init_logger(log_dir)

        self.input_dir = Path(input_dir)
        self.chunks_in_input_dir = chunks_in_input_dir

        self.utterance_lookup: dict[int, UtteranceInfo] = {}
        self.new_utterance_by_frame = defaultdict(list)

        with open(self.input_dir / self.LOG_FILE, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i:j for i,j in zip(keys, row)}
                utterance = UtteranceInfo(
                        int(data["utterance_id"]),
                        int(data["frame_received"]),
                        data["speaker_id"],
                        data["text"],
                        int(data["start_frame"]),
                        int(data["stop_frame"]),
                        self.get_chunk_file(data["audio_file"])
                    )
                self.utterance_lookup[utterance.utterance_id] = utterance
                self.new_utterance_by_frame[utterance.frame_received].append(utterance.utterance_id)

        self.current_frame = 0

    def exit(self):
        pass

    def get_chunk_file(self, file):
        if not self.chunks_in_input_dir:
            return file
        else:
            chunk_name = Path(file).name
            return str(self.input_dir / "chunks" / chunk_name)

    def processFrame(self, frame, frame_count, time_to_frame, includeText):
        new_utterance_ids = []
        while self.current_frame <= frame_count:
            new_utterance_ids += self.new_utterance_by_frame[self.current_frame]
            self.current_frame += 1

        for i in new_utterance_ids:
            utterance = self.utterance_lookup[i]
            self.log_utterance(utterance)

        if includeText:
            cv2.putText(frame, "ASR evaluation", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return new_utterance_ids
