import torch
import torch.nn.functional as F
from featureModules.move.move_classifier import (
    rec_common_ground,
    hyperparam,
    modalities,
)
from featureModules.move.closure_rules import CommonGround
from transformers import BertTokenizer, BertModel
import cv2
import opensmile
import re

from logger import Logger

# length of the sequence (the utterance of interest + 3 previous utterances for context)
UTTERANCE_HISTORY_LEN = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("move classifier device", device)

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

class MoveFeature:
    def __init__(self, txt_log_file=None):
        self.model = torch.load(r"featureModules\move\production_move_classifier.pt").to(device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model: BertModel = BertModel.from_pretrained("bert-base-uncased").to(
            device
        )  # pyright: ignore

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.bert_embedding_history = torch.zeros(
            (UTTERANCE_HISTORY_LEN, 768), device=device
        )
        self.opensmile_embedding_history = torch.zeros(
            (UTTERANCE_HISTORY_LEN, 88), device=device
        )

        self.closure_rules = CommonGround()
        self.class_names = ["STATEMENT", "ACCEPT", "DOUBT"]

        self.most_recent_prop = "no prop"

        self.logger = Logger(file=txt_log_file, stdout=True)
        self.logger.clear()


    def update_bert_embeddings(self, name, text):
        input_ids = torch.tensor(self.tokenizer.encode(text), device=device).unsqueeze(0)
        cls_embeddings = self.bert_model(input_ids)[0][:, 0]

        self.bert_embedding_history = torch.cat([self.bert_embedding_history[1:], cls_embeddings])

    def update_smile_embeddings(self, name, audio_file):
        embedding = torch.tensor(self.smile.process_file(audio_file).to_numpy(), device=device)

        self.opensmile_embedding_history = torch.cat([self.opensmile_embedding_history[1:], embedding])

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
            for i, line in enumerate(labels):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                y = ((int(blockHeight / 4) + int(th / 2)) * (i + 1)) + p2
                x = (int(blockWidth / 2) - int(tw / 2)) + p1
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    def processFrame(self, utterances_and_props, frame, frameIndex, includeText, banks):
        for name, text, prop, audio_file in utterances_and_props:
            if prop != "no prop":
                self.most_recent_prop = prop

            self.update_bert_embeddings(name, text)
            in_bert = self.bert_embedding_history

            self.update_smile_embeddings(name, audio_file)
            in_open = self.opensmile_embedding_history

            # TODO: other inputs for move classifier
            in_cps = torch.zeros((UTTERANCE_HISTORY_LEN, 3), device=device)
            in_action = torch.zeros((UTTERANCE_HISTORY_LEN, 78), device=device)
            in_gamr = torch.zeros((UTTERANCE_HISTORY_LEN, 243), device=device)

            # out = F.softmax(self.model(in_bert, in_open, in_cps, in_action, in_gamr, hyperparam, modalities))
            out = torch.sigmoid(self.model(in_bert, in_open, in_cps, in_action, in_gamr, hyperparam, modalities))
            out = out.cpu().detach().numpy()

            present_class_indices = (out > 0.5)
            move = [self.class_names[idx] for idx, class_present in enumerate(present_class_indices) if class_present]

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

        if banks:
            self.renderBanks(frame, 130, 260, "F BANK:", self.closure_rules.fbank)
            self.renderBanks(frame, 130, 130, "E BANK:",  self.closure_rules.ebank)
        if includeText:
            cv2.putText(frame, "Move classifier is live", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
