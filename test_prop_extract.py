from featureModules.prop.demo import process_sentence, load_model
from featureModules.prop.PropExtractFeature import COLORS, NUMBERS


model_dir = r'featureModules\prop\data\prop_extraction_model'
model, tokenizer = load_model(model_dir)

while True:
    text = input("enter phrase: ")
    contains_color = any(i in text for i in COLORS)
    contains_number = any(i in text for i in NUMBERS)
    if contains_color or contains_number:
        prop, num_filtered_props = process_sentence(text, model, tokenizer, verbose=False)
    else:
        prop, num_filtered_props = "no prop", 0
    print("prop: ", prop)
    
