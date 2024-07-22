from faster_whisper import tokenizer
from featureModules.IFeature import *
from logger import Logger
from utils import *
from featureModules.prop.demo import process_sentence, load_model

class PropExtractFeature(IFeature):
    def __init__(self, csv_log_file=None):
        model_dir = r'featureModules\prop\data\prop_extraction_model'
        self.model, self.tokenizer = load_model(model_dir)
        self.logger = Logger(file=csv_log_file)
        self.logger.write_csv_headers("frame", "name", "text", "prop")

    def processFrame(self, frame, utterances, frame_count, includeText):
        utterances_and_props = []
        for name, start, stop, text, audio_file in utterances:
            colors = ["red", "blue", "green", "purple", "yellow"]
            numbers = ["10", "20", "30", "40", "50"]
            if not any(i in text for i in colors + numbers):
                prop = "no prop"
            else:
                prop = process_sentence(text, self.model, self.tokenizer, verbose=False)

            utterances_and_props.append((name, text, prop, audio_file))
            self.logger.append_csv(frame_count, name, text, prop)

        if includeText:
            cv2.putText(frame, "Prop extract is live", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return utterances_and_props
