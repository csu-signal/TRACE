from dataclasses import dataclass
from featureModules.IFeature import *
from featureModules.asr.AsrFeature import UtteranceInfo
from logger import Logger
from utils import *
from featureModules.prop.demo import process_sentence, load_model
from sentence_transformers import SentenceTransformer
from featureModules.prop.demoHelpers import get_cg_embeddings, get_pickle
from sentence_transformers import SentenceTransformer
from featureModules.prop.demoHelpers import get_cg_embeddings, get_pickle

COLORS = ["red", "blue", "green", "purple", "yellow"]
NUMBERS = ["10", "20", "30", "40", "50"]

@dataclass
class PropInfo:
    utterance_id: int
    prop: str

class PropExtractFeature(IFeature):
    LOG_FILE = "propOutput.csv"

    def __init__(self, log_dir=None):
        model_dir = r'featureModules\prop\data\prop_extraction_model'
        self.model, self.tokenizer = load_model(model_dir)
        self.bert = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
        self.embeddings = get_pickle()        self.bert = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
        self.embeddings = get_pickle()
        self.init_logger(log_dir)

        # map utterance ids to propositions
        self.prop_lookup = {}

    def init_logger(self, log_dir):
        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.write_csv_headers("frame", "utterance_id", "proposition", "utterance_text", "num_filtered_props")

    def log_prop(self, frame_count, prop_info: PropInfo, text, num_props):
        self.logger.append_csv(frame_count, prop_info.utterance_id, prop_info.prop, text, num_props)

    def processFrame(self, frame, new_utterance_ids: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo], frame_count: int, includeText):
        for i in new_utterance_ids:
            utterance_info = utterance_lookup[i]

            contains_color = any(i in utterance_info.text for i in COLORS)
            contains_number = any(i in utterance_info.text for i in NUMBERS)
            if contains_color or contains_number:
                prop, num_filtered_props = process_sentence(utterance_info.text, self.model, self.tokenizer, self.bert, self.embeddings, verbose=False)
            else:
                prop, num_filtered_props = "no prop", 0

            self.prop_lookup[i] = PropInfo(i, prop)
            self.log_prop(frame_count, self.prop_lookup[i], utterance_info.text, num_filtered_props)

        if includeText:
            cv2.putText(frame, "Prop extract is live", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

