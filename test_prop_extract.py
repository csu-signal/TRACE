from featureModules.prop.demo import process_sentence, load_model


model_dir = r'featureModules\prop\data\prop_extraction_model'
model, tokenizer = load_model(model_dir)

while True:
    text = input("enter phrase: ")
    colors = ["red", "blue", "green", "purple", "yellow"]
    numbers = ["10", "20", "30", "40", "50"]
    if not any(i in text for i in colors + numbers):
        prop, num_filtered_props = "no prop", 0
    else:
        prop, num_filtered_props = process_sentence(text, model, tokenizer, verbose=False)
    print("prop: ", prop)
    
