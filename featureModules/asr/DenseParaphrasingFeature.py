from featureModules.IFeature import IFeature
from featureModules.asr.AsrFeature import UtteranceInfo

import re

from logger import Logger

class Demonstrative():
    def __init__(self, regex, plural):
        self.regex = regex
        self.plural = plural

demonstratives = [
    Demonstrative(r"\bthose\b", True), 
    Demonstrative(r"\bthese\b", True), 
    Demonstrative(r"\bthis\b", False), 
    Demonstrative(r"\bthat\b", False), 
    Demonstrative(r"\bit\b", False)]

class DenseParaphrasingFeature(IFeature):
    LOG_FILE = "dense_paraphrasing_out.csv"

    def __init__(self, log_dir=None):
        self.paraphrased_utterance_lookup: dict[int, UtteranceInfo] = {}

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()
        self.logger.write_csv_headers("frame", "utterance_id", "updated_text", "old_text", "subs_made")

    def processFrame(self, frame, new_utterances: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo],
                             blockCache, frame_count):
        clear = False
        for i in new_utterances:
            utterance_info = utterance_lookup[i]
            text = utterance_info.text

            for demo in demonstratives:
                demos = re.finditer(demo.regex, text)
                spanList = [d.span() for d in [*demos]]
                demoCount = len(spanList)

                if (demoCount > 0):
                    key = int(utterance_info.start)
                    while(key < utterance_info.stop):
                        if key in blockCache:
                            targets = blockCache[key]
                            targetList = [t.description for t in targets]

                            if (demoCount == 1):
                                if (not demo.plural):
                                    targetList = targetList[0]
                                    text = re.sub(demo.regex, targetList, text.lower())
                                else:
                                    text = re.sub(demo.regex, ', '.join(targetList), text.lower())
                            else:
                                for i in range(len(targetList)):
                                    if i < len(spanList):
                                        text = re.sub(demo.regex, targetList[i], text.lower()[:spanList[i][1]]) + text.lower()[spanList[i][1]:]
                            break

                        key+=1

            self.paraphrased_utterance_lookup[i] = UtteranceInfo(
                    i,
                    frame_count,
                    utterance_info.speaker_id,
                    utterance_info.text,
                    utterance_info.start,
                    utterance_info.stop,
                    utterance_info.audio_file
                )

            self.logger.append_csv(
                    frame_count,
                    i,
                    text,
                    utterance_info.text,
                    text.lower() != utterance_info.text.lower()
                )

        #TODO when should we clear these cached values?  
        # if(clear):
        #     self.blockCache = {} 
