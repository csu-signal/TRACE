from faster_whisper import tokenizer
from featureModules.IFeature import *
from utils import *
from featureModules.prop.demo import process_sentence, load_model

class PropExtractFeature(IFeature):
    def __init__(self):
        model_dir = r'featureModules\prop\data\prop_extraction_model'
        self.model, self.tokenizer = load_model(model_dir, verbose=True)

    def processFrame(self, frame, utterances):
        utterances_and_props = []
        for name, start, stop, text, audio_file in utterances:
            prop = process_sentence(text, self.model, self.tokenizer, verbose=False)

            # TODO: make a better way to signal failure
            if prop == "FAILURE":
                prop = "no prop"
            utterances_and_props.append((name, text, prop, audio_file))

        cv2.putText(frame, "Prop extract is live", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return utterances_and_props
