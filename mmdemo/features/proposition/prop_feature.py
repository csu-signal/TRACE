from pathlib import Path
from typing import final

import cv2
from sentence_transformers import SentenceTransformer

from mmdemo.base_feature import BaseFeature
from mmdemo.features.proposition.demo import load_model, process_sentence
from mmdemo.features.proposition.demo_helpers import get_pickle
from mmdemo.features.proposition.prop_info import PropInfo
from mmdemo.interfaces import PropositionInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...

COLORS = ["red", "blue", "green", "purple", "yellow"]
NUMBERS = ["10", "20", "30", "40", "50"]


@final
class Proposition(BaseFeature[PropositionInterface]):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_input_interfaces(cls):
        return [
            TranscriptionInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return PropositionInterface

    def initialize(self):
        model_dir = str(Path(__file__).parent / "data/prop_extraction_model")
        self.model, self.tokenizer = load_model(model_dir)
        self.bert = SentenceTransformer(
            "sentence-transformers/multi-qa-distilbert-cos-v1"
        )
        self.embeddings = get_pickle(self.bert)

        # map utterance ids to propositions
        self.prop_lookup = {}
        pass

    def get_output(
        self,
        tran: TranscriptionInterface,
    ):
        if not tran.is_new():
            return None
        for i in tran.start_time:
            utterance_info = tran.text[i]

            contains_color = any(i in utterance_info.text for i in COLORS)
            contains_number = any(i in utterance_info.text for i in NUMBERS)
            if contains_color or contains_number:
                prop, num_filtered_props = process_sentence(
                    utterance_info.text,
                    self.model,
                    self.tokenizer,
                    self.bert,
                    self.embeddings,
                    verbose=False,
                )
            else:
                prop, num_filtered_props = "no prop", 0

            self.prop_lookup[i] = PropInfo(i, prop)
            self.log_prop(
                frame_count,
                self.prop_lookup[i],
                utterance_info.text,
                num_filtered_props,
            )

        if includeText:
            cv2.putText(
                frame,
                "Prop extract is live",
                (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        # call prop extractor, create interface, and return

    def log_prop(self, frame_count, prop_info: PropInfo, text, num_props):
        self.logger.append_csv(
            frame_count, prop_info.utterance_id, prop_info.prop, text, num_props
        )

    def processFrame(
        self,
        frame,
        new_utterance_ids: list[int],
        utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo],
        frame_count: int,
        includeText,
    ):
        for i in new_utterance_ids:
            utterance_info = utterance_lookup[i]

            contains_color = any(i in utterance_info.text for i in COLORS)
            contains_number = any(i in utterance_info.text for i in NUMBERS)
            if contains_color or contains_number:
                prop, num_filtered_props = process_sentence(
                    utterance_info.text,
                    self.model,
                    self.tokenizer,
                    self.bert,
                    self.embeddings,
                    verbose=False,
                )
            else:
                prop, num_filtered_props = "no prop", 0

            self.prop_lookup[i] = PropInfo(i, prop)
            self.log_prop(
                frame_count,
                self.prop_lookup[i],
                utterance_info.text,
                num_filtered_props,
            )

        if includeText:
            cv2.putText(
                frame,
                "Prop extract is live",
                (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
