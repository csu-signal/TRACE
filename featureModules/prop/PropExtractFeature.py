from faster_whisper import tokenizer
from featureModules.IFeature import *
from utils import *
from featureModules.prop.demo import process_sentence, load_model

class PropExtractFeature(IFeature):
    def __init__(self):
        model_dir = r'featureModules\prop\data\prop_extraction_model'
        self.model, self.tokenizer = load_model(model_dir, verbose=True)

    def processFrame(self, frame, utterances):
        for name, start, stop, text in utterances:
            prop = process_sentence(text, self.model, self.tokenizer, verbose=False)
            print(f"{name}: {text} ==> {prop}")

        cv2.putText(frame, "Prop extract is live", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
