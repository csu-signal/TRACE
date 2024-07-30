from demo.featureModules.prop.demo import process_sentence, load_model
from demo.featureModules.prop.PropExtractFeature import COLORS, NUMBERS
from sentence_transformers import SentenceTransformer
from demo.featureModules.prop.demoHelpers import *
from sentence_transformers import SentenceTransformer
from demo.featureModules.prop.demoHelpers import *


model_dir = r'featureModules\prop\data\prop_extraction_model'
model, tokenizer = load_model(model_dir)
bert = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
embeddings = get_pickle(bert)

while True:
    text = input("enter phrase: ")
    contains_color = any(i in text for i in COLORS)
    contains_number = any(i in text for i in NUMBERS)
    if contains_color or contains_number:
        prop, num_filtered_props = process_sentence(text, model, tokenizer, bert, embeddings, verbose=False)
    else:
        prop, num_filtered_props = "no prop", 0
    print("prop: ", prop)
    
