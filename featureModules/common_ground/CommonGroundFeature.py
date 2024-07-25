from featureModules.IFeature import IFeature
from featureModules.common_ground.closure_rules import CommonGround
from featureModules.move.MoveFeature import MoveInfo
from featureModules.prop.PropExtractFeature import PropInfo
from logger import Logger
import cv2
import re

class Color():
    def __init__(self, name, color):
        self.name = name
        self.color = color

colors = [
        Color("red", (0, 0, 255)), 
        Color("blue", (255, 0, 0)), 
        Color("green", (19, 129, 51)), 
        Color("purple", (128, 0, 128)), 
        Color("yellow", (0, 215, 255))]

fontScales = [1.5, 1.5, 0.75, 0.5, 0.5]
fontThickness = [3, 3, 2, 2, 2]

class CommonGroundFeature(IFeature):
    LOG_FILE = "common_ground_output.csv"

    def __init__(self, log_dir=None):
        self.closure_rules = CommonGround()

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.write_csv_headers("frame", "utterance_id", "qbank", "ebank", "fbank", "prop", "move")

        self.most_recent_prop = "no prop"

    def getPropValues(self, propStrings, match):
        label = []
        for prop in propStrings:
            prop_match = re.match(r'(' + match + r')\s*(=|<|>|!=)\s*(.*)', prop)
            if prop_match:
                block = prop_match[1]
                relation = prop_match[2]
                rhs = prop_match[3]
                if(relation == '<' or relation == '>' or relation == '!='):
                    label.append(relation + rhs)
                else:
                    label.append(rhs)
        return label

    def renderBanks(self, frame, xSpace, yCord, bankLabel, bankValues):
        blocks = len(colors) + 1
        blockWidth = 112
        blockHeight = 112

        h,w,_ = frame.shape
        start = w - (xSpace * blocks)
        p2 = h - yCord
        (tw, th), _ = cv2.getTextSize(bankLabel, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        labelCoords = (int(start) - int(tw / 3), (int(blockHeight / 2) + int(th / 2)) + p2)
        cv2.putText(frame, bankLabel, labelCoords, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

        for i in range(1, blocks):
            p1 = start + (xSpace * i)
            color = colors[i - 1]
            cv2.rectangle(frame, 
                (p1, p2), 
                (p1 + blockWidth, p2 + blockHeight), 
                color=color.color,
                thickness=-1)
            
            labels = self.getPropValues(bankValues, color.name)
            numberLabels = min(len(labels), 5)
            if(numberLabels > 0):
                for i, line in enumerate(labels):
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fontScales[numberLabels - 1], fontThickness[numberLabels -1])
                    y = ((int(blockHeight / (numberLabels + 1)) + int(th / 3)) * (i + 1)) + p2
                    x = (int(blockWidth / 2) - int(tw / 2)) + p1
                    cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScales[numberLabels - 1], (0,0,0), fontThickness[numberLabels -1])

    # TODO: create moveinfo
    def processFrame(self, frame, new_utterances: list[int], prop_lookup: dict[int, PropInfo], move_lookup: dict[int, MoveInfo], frame_count):
        for i in new_utterances:
            prop = prop_lookup[i].prop
            move = move_lookup[i].move

            if prop != "no prop":
                self.most_recent_prop = prop

            self.closure_rules.update(move, self.most_recent_prop)

            self.logger.append_csv(frame_count, i, self.closure_rules.qbank, self.closure_rules.ebank, self.closure_rules.fbank, prop, move)


        self.renderBanks(frame, 130, 260, "F BANK:", self.closure_rules.fbank)
        self.renderBanks(frame, 130, 130, "E BANK:",  self.closure_rules.ebank)
