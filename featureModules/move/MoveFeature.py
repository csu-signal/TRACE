import torch
import torch.nn.functional as F
from featureModules.asr.AsrFeature import UtteranceInfo
from featureModules.move.move_classifier import (
    rec_common_ground,
    hyperparam,
    modalities,
)
from transformers import BertTokenizer, BertModel
import cv2
import opensmile

from logger import Logger

# length of the sequence (the utterance of interest + 3 previous utterances for context)
UTTERANCE_HISTORY_LEN = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("move classifier device", device)


class MoveFeature:
    LOG_FILE = "moveOutput.txt"

    def __init__(self, log_dir=None):
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

        self.class_names = ["STATEMENT", "ACCEPT", "DOUBT"]

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.clear()


    def update_bert_embeddings(self, name, text):
        input_ids = torch.tensor(self.tokenizer.encode(text), device=device).unsqueeze(0)
        cls_embeddings = self.bert_model(input_ids)[0][:, 0]

        self.bert_embedding_history = torch.cat([self.bert_embedding_history[1:], cls_embeddings])

    def update_smile_embeddings(self, name, audio_file):
        embedding = torch.tensor(self.smile.process_file(audio_file).to_numpy(), device=device)

        self.opensmile_embedding_history = torch.cat([self.opensmile_embedding_history[1:], embedding])

    def processFrame(self, frame, new_utterances: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo], frameIndex):
        for name, text, prop, audio_file in utterances_and_props:
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


        cv2.putText(frame, "Move classifier is live", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
