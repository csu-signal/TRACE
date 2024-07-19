from featureModules.IFeature import IFeature
from featureModules.asr.AsrFeature import UtteranceInfo
from featureModules.common_ground.closure_rules import CommonGround
from featureModules.prop.PropExtractFeature import PropInfo
from logger import Logger

class CommonGroundFeature(IFeature):
    LOG_FILE = "common_ground_output.csv"

    def __init__(self, log_dir=None):
        self.closure_rules = CommonGround()

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.write_csv_headers("frame", "utterance_id", "qbank", "ebank", "fbank")

        self.most_recent_prop = "no prop"

    # TODO: create moveinfo
    def processFrame(self, new_utterances: list[int], prop_lookup: dict[int, PropInfo], move_lookup: dict[int, MoveInfo], frame_count):
        pass
        for i in new_utterances:
            prop = prop_lookup[i].prop
            move = move_lookup[i].move

            if prop != "no prop":
                self.most_recent_prop = prop

            self.closure_rules.update(move, self.most_recent_prop)
            update = ""
            update += "FRAME: " + str(frameIndex) + "\n"
            update += "Q bank\n"
            update += str(self.closure_rules.qbank) + "\n"
            update += "E bank\n"
            update += str(self.closure_rules.ebank) + "\n"
            update += "F bank\n"
            update += str(self.closure_rules.fbank) + "\n"
            if prop == "no prop":
                update += f"{name}: {text} ({self.most_recent_prop}), {out}\n\n"
            else:
                update += f"{name}: {text} => {self.most_recent_prop}, {out}\n\n"

            self.logger.append(update)
