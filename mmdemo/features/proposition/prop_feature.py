from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ASRInterface,
    BodyTrackingInterface,
    ColorImageInterface,
    DenseParaphraseInterface,
    DepthImageInterface,
    GestureInterface,
    ObjectInterface,
    PropositionInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
    UtteranceChunkInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Proposition(BaseFeature):
    # LOG_FILE = "propOutput.csv"
    @classmethod
    def get_input_interfaces(cls):
        return [
            DenseParaphraseInterface,
            SelectedObjectsInterface,
            ObjectInterface,
            GestureInterface,
            ColorImageInterface,
            DepthImageInterface,
            BodyTrackingInterface,
            ASRInterface,
            UtteranceChunkInterface,
            TranscriptionInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return PropositionInterface

    def initialize(self):
        # self.model, self.tokenizer = load_model(model_dir)
        # self.bert = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
        # self.init_logger(log_dir)
        # self.embeddings = get_pickle(self.bert)

        # # map utterance ids to propositions
        # self.prop_lookup = {}
        pass

    def get_output(
        self,
        dense: DenseParaphraseInterface,
        select_obj: SelectedObjectsInterface,
        obj: ObjectInterface,
        gest: GestureInterface,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        bod: BodyTrackingInterface,
        asr: ASRInterface,
        utt: UtteranceChunkInterface,
        tran: TranscriptionInterface,
    ):
        if (
            not dense.is_new()
            and not select_obj.is_new()
            and not obj.is_new()
            and not gest.is_new()
            and not col.is_new()
            and not dep.is_new
            and not bod.is_new()
            and not asr.is_new()
            and not utt.is_new()
            and not tran.is_new()
        ):
            return None

        # call prop extractor, create interface, and return

    # def init_logger(self, log_dir):
    #     if log_dir is not None:
    #         self.logger = Logger(file=log_dir / self.LOG_FILE)
    #     else:
    #         self.logger = Logger()

    #     self.logger.write_csv_headers("frame", "utterance_id", "proposition", "utterance_text", "num_filtered_props")

    # def log_prop(self, frame_count, prop_info: PropInfo, text, num_props):
    #     self.logger.append_csv(frame_count, prop_info.utterance_id, prop_info.prop, text, num_props)

    # def processFrame(self, frame, new_utterance_ids: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo], frame_count: int, includeText):
    #     for i in new_utterance_ids:
    #         utterance_info = utterance_lookup[i]

    #         contains_color = any(i in utterance_info.text for i in COLORS)
    #         contains_number = any(i in utterance_info.text for i in NUMBERS)
    #         if contains_color or contains_number:
    #             prop, num_filtered_props = process_sentence(utterance_info.text, self.model, self.tokenizer, self.bert, self.embeddings, verbose=False)
    #         else:
    #             prop, num_filtered_props = "no prop", 0

    #         self.prop_lookup[i] = PropInfo(i, prop)
    #         self.log_prop(frame_count, self.prop_lookup[i], utterance_info.text, num_filtered_props)

    #     if includeText:
    #         cv2.putText(frame, "Prop extract is live", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)