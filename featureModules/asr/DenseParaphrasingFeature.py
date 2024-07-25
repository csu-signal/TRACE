from featureModules.IFeature import IFeature
from featureModules.asr.AsrFeature import UtteranceInfo

import re

from logger import Logger

class Demonstrative():
    def __init__(self, text, plural):
        self.text = text
        self.plural = plural

demonstratives = [
    Demonstrative("those", True), 
    Demonstrative("these", True), 
    Demonstrative("this", False), 
    Demonstrative("that", False), 
    Demonstrative("it", False)]

class DenseParaphrasingFeature(IFeature):
    LOG_FILE = "dense_paraphrasing_out.csv"

    def __init__(self, log_dir=None):
        self.paraphrased_utterance_lookup: dict[int, UtteranceInfo] = {}

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()
        self.logger.write_csv_headers("frame", "utterance_id", "updated_text", "old_text", "subs_made")

    def processFrame(self, frame, new_utterances: list[int], utterance_lookup: dict[int, UtteranceInfo], blockCache, frame_count):
        clear = False
        for i in new_utterances:
            utterance_info = utterance_lookup[i]
            text = utterance_info.text

            plural_demo_regex = r"\b(" + "|".join([d.text for d in demonstratives if d.plural]) + r")\b"
            singlular_demo_regex = r"\b(" + "|".join([d.text for d in demonstratives if not d.plural]) + r")\b"

            key = int(utterance_info.start)
            while key < utterance_info.stop:
                if key in blockCache and len(blockCache[key]) > 0:
                    targets = blockCache[key]

                    text = re.sub(plural_demo_regex, ", ".join(targets), text, count=1, flags=re.IGNORECASE)

                    for i in range(len(targets)):
                        text = re.sub(singlular_demo_regex, targets[i], text, count=1, flags=re.IGNORECASE)

                key+=1

            self.paraphrased_utterance_lookup[utterance_info.utterance_id] = UtteranceInfo(
                    utterance_info.utterance_id,
                    frame_count,
                    utterance_info.speaker_id,
                    text,
                    utterance_info.start,
                    utterance_info.stop,
                    utterance_info.audio_file
                )

            self.logger.append_csv(
                    frame_count,
                    utterance_info.utterance_id,
                    text,
                    utterance_info.text,
                    text.lower() != utterance_info.text.lower()
                )

        #TODO when should we clear these cached values?  
        # if(clear):
        #     self.blockCache = {} 
