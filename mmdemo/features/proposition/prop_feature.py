from pathlib import Path
from typing import final

from sentence_transformers import SentenceTransformer

from mmdemo.base_feature import BaseFeature
from mmdemo.features.proposition.demo import load_model, process_sentence
from mmdemo.features.proposition.demo_helpers import get_pickle
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface

COLORS = ["red", "blue", "green", "purple", "yellow"]
NUMBERS = ["10", "20", "30", "40", "50"]


@final
class Proposition(BaseFeature[PropositionInterface]):
    # TODO: docstring
    def initialize(self):
        model_dir = str(Path(__file__).parent / "data/prop_extraction_model")
        self.model, self.tokenizer = load_model(model_dir)
        self.bert = SentenceTransformer(
            "sentence-transformers/multi-qa-distilbert-cos-v1"
        )
        self.embeddings = get_pickle(self.bert)

    def get_output(
        self,
        tran: TranscriptionInterface,
    ):
        if not tran.is_new():
            return None

        contains_color = any(i in tran.text for i in COLORS)
        contains_number = any(i in tran.text for i in NUMBERS)
        if contains_color or contains_number:
            prop, num_filtered_props = process_sentence(
                tran.text,
                self.model,
                self.tokenizer,
                self.bert,
                self.embeddings,
                verbose=False,
            )
        else:
            prop, num_filtered_props = "no prop", 0

        return PropositionInterface(speaker_id=tran.speaker_id, prop=prop)
