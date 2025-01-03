import pickle
from pathlib import Path
from typing import final


import opensmile
import torch
from transformers import BertModel, BertTokenizer, PreTrainedModel
from joblib import load

from mmdemo.base_feature import BaseFeature
from mmdemo.features.move.move_classifier import (
    hyperparam,
    modalities,
    rec_common_ground,
)
from mmdemo.interfaces import AudioFileInterface, MoveInterface, TranscriptionInterface, GestureConesInterface, SelectedObjectsInterface

UTTERANCE_HISTORY_LEN = 4
BERT_EMBEDDING_DIM = 768
SMILE_EMBEDDING_DIM = 88
GAMR_EMBEDDING_DIM = 128


class custom_pickle:
    """
    The torch model requires a class called `rec_common_ground` to be in the module
    `__main__`. This means that it needs to be imported in every file where the demo
    runs, which is very inconvenient. This class essentially tricks pickle into using
    the local `rec_common_ground` instead.
    """

    class Unpickler(pickle.Unpickler):
        def find_class(self, module_name: str, global_name: str):
            if module_name == "__main__" and global_name == "rec_common_ground":
                return rec_common_ground
            return super().find_class(module_name, global_name)


@final
class Move(BaseFeature[MoveInterface]):
    """
    Determine moves of participants (statement, accept, doubt)

    Input interfaces are `TranscriptionInterface`, `AudioFileInterface`
    Output interface is `MoveInterface`

    Keyword arguments:
    `model_path` -- the path to the model (or None to use the default)
    """

    DEFAULT_MODEL_PATH = Path(__file__).parent / "production_move_classifier.pt"

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        audio: BaseFeature[AudioFileInterface],
        gesture: BaseFeature[GestureConesInterface]|None,
        objects: BaseFeature[SelectedObjectsInterface]|None,
        *,
        model_path: Path | None = None
    ) -> None:
        if gesture is None and objects is None:
            super().__init__(transcription, audio)
        if gesture is not None and objects is not None:
            super().__init__(transcription, audio, gesture, objects)
        if gesture is not None and objects is None:
            super().__init__(transcription, audio, gesture)

        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path

    def initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.load(
            str(self.model_path), pickle_module=custom_pickle, map_location=self.device
        )
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model: PreTrainedModel = BertModel.from_pretrained(
            "bert-base-uncased"
        ).to(self.device)

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.bert_embedding_history = torch.zeros(
            (UTTERANCE_HISTORY_LEN, BERT_EMBEDDING_DIM), device=self.device
        )
        self.opensmile_embedding_history = torch.zeros(
            (UTTERANCE_HISTORY_LEN, SMILE_EMBEDDING_DIM), device=self.device
        )
        self.gamr_embedding_history = torch.zeros(
            (UTTERANCE_HISTORY_LEN, GAMR_EMBEDDING_DIM), device=self.device
        )

        self.class_names = ["STATEMENT", "ACCEPT", "DOUBT"]

    def get_output(
        self,
        transcription: TranscriptionInterface,
        audio: AudioFileInterface,
        gesture: GestureConesInterface | None = GestureConesInterface(wtd_body_ids=[], azure_body_ids=[], handedness=[], cones=[]),
        objects: SelectedObjectsInterface | None = None,
    ):
        if not transcription.is_new() or not audio.is_new():
            return None

        text = transcription.text
        audio_file = audio.path

        self.update_bert_embeddings(text)
        in_bert = self.bert_embedding_history

        self.update_smile_embeddings(audio_file)
        in_open = self.opensmile_embedding_history

        # TODO: other inputs for move classifier
        in_cps = torch.zeros((UTTERANCE_HISTORY_LEN, 3), device=self.device)
        in_action = torch.zeros((UTTERANCE_HISTORY_LEN, 78), device=self.device)
        in_gamr = torch.zeros((UTTERANCE_HISTORY_LEN, 243), device=self.device)

        out = torch.sigmoid(
            self.model(
                in_bert, in_open, in_cps, in_action, in_gamr, hyperparam, modalities
            )
        )
        out = out.cpu().detach().numpy()

        present_class_indices = out > 0.5
        move = [
            self.class_names[idx]
            for idx, class_present in enumerate(present_class_indices)
            if class_present
        ]

        return MoveInterface(speaker_id=transcription.speaker_id, move=move)

    def update_bert_embeddings(self, text):
        input_ids = torch.tensor(
            self.tokenizer.encode(text), device=self.device
        ).unsqueeze(0)
        cls_embeddings = self.bert_model(input_ids)[0][:, 0]

        self.bert_embedding_history = torch.cat(
            [self.bert_embedding_history[1:], cls_embeddings]
        )

    def update_smile_embeddings(self, audio_file):
        embedding = torch.tensor(
            self.smile.process_file(audio_file).to_numpy(), device=self.device
        )
        self.opensmile_embedding_history = torch.cat(
            [self.opensmile_embedding_history[1:], embedding]
        )

