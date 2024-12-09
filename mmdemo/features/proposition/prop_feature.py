from pathlib import Path
from typing import final

import nltk
from sentence_transformers import SentenceTransformer

from mmdemo.base_feature import BaseFeature
from mmdemo.features.proposition.demo import load_model, process_sentence
from mmdemo.features.proposition.demo_helpers import get_pickle
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface

COLORS = ["red", "blue", "green", "purple", "yellow"]
NUMBERS = ["10", "20", "30", "40", "50"]


@final
class Proposition(BaseFeature[PropositionInterface]):
    """
    Extract propositions from a transcription.

    Input interface is `TranscriptionInterface`

    Output interface is `PropositionInterface`

    Keyword arguments:
    `model_path` -- the path to the model (or None to use the default)
    """

    DEFAULT_MODEL_PATH = Path(__file__).parent / "data/prop_extraction_model"

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        *,
        model_path: Path | None = None
    ):
        super().__init__(transcription)
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path

    def initialize(self):
        self.model, self.tokenizer = load_model(str(self.model_path))
        self.bert = SentenceTransformer(
            "sentence-transformers/multi-qa-distilbert-cos-v1"
        )
        self.embeddings = get_pickle(self.bert)
        nltk.download("stopwords")
        nltk.download("punkt_tab")

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
        print(f"prop is : {prop}")
        return PropositionInterface(speaker_id=tran.speaker_id, prop=prop)
