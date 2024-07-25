from demo import process_sentence, load_model, remove_stop_words


model_dir = r'/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/XE/googleSandbox/XE_models/model'
model, tokenizer = load_model(model_dir)
'''
while True:
    text = input("enter phrase: ")
    colors = ["red", "blue", "green", "purple", "yellow"]
    numbers = ["10", "20", "30", "40", "50"]
    # Check for presence of colors and numbers
    contains_color = any(color in text for color in colors)
    contains_number = any(number in text for number in numbers)

    # Determine action based on presence of colors and numbers
    if not (contains_color or contains_number):
        prop, num_filtered_props = "no prop", 0
    elif contains_color and not contains_number:
        prop, num_filtered_props = "no prop", 0
    elif contains_number and not contains_color:
        prop, num_filtered_props = "no prop", 0
    else:
        prop, num_filtered_props = process_sentence(text, model, tokenizer, verbose=False)
    print("prop: ", prop)
    
'''

listOfProp = ["blue is not 10",
              "I think yellow is definetely not 40.", 
              " Red 10",
              "It seems like blue might be about the same as the red",
              " Green looks like about 20.",
              "So, yellow block is noticeably heavier than the purple one.",
              "So red is a 10 and green is a 20 right there ",
              "I would say purple about 30",
              "Wait, lets make sure purple is not also 20 ",
              "so Purple is more than 20.",
              "Purple has to be 30",
              "I would guess the yellow would be  equal to 40 ",
              "Did we say purple was 30 ",
              "Yeah, purple is 30, green is 20, and blue and red are both 10.",
              "the yellow is not 40",
              "Yeah, green plus purple is yellow ",
              "purple is more than 20",
              "purple is not 20",
              "Yeah, we did try yellow as 40 because purple block is 30",
              "So, purple is 30, and blue is 10? ",
              "So, yellow is at least closest to 50",
              "Green looks like about 20",
              "Blue is a 10 and green is a 20",
              "Lets double check that purple is also not 20",
              "I would say that purple is about 30",
              "Blue is the same as the red", 
              "Yellow is noticebly heavier than purple",
              "Yellow is definitely not 40, it's too heavy.",
              "30"
              ]

#print(remove_stop_words(" Green looks like about 20."))
for sentences in listOfProp:
    
    prop, num_filtered_props = process_sentence(sentences, model, tokenizer, verbose=False)
    print("UTT: ", sentences, "CLEAN:" , remove_stop_words(sentences) , "PROP: ", prop)
    
