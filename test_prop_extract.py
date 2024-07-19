from featureModules.prop.demo import process_sentence, load_model


model_dir = r'featureModules\prop\data\prop_extraction_model'
model, tokenizer = load_model(model_dir)

while True:
    text = input("enter phrase: ")
    output = process_sentence(text, model, tokenizer, verbose=False)
    print("prop: ", output)
    
