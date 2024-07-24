import torch
import pandas as pd
#from featureModules.prop.models import CrossEncoder
from featureModules.prop.models import CrossEncoder
import string
from transformers import AutoTokenizer

from featureModules.prop.demoHelpers import *
# from featureModules.prop.demoHelpers import tokenize_props, extract_colors_and_numbers, is_valid_common_ground, \
# is_valid_individual_match, predict_with_XE, add_special_tokens, get_embeddings, sentence_fcg_cosine
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk import word_tokenize, download
from nltk.corpus import stopwords
download("stopwords")
download("punkt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def remove_stop_words(utterance):
    #default stopwords 
    stop_words = set(stopwords.words('english'))
    utterance = utterance.lower()
    #custom stopwords
    # additional_stop_words = ['so', 'yeah', 'well', 'uh', 'ok', 'now', 'we', 'know', 'that', 'we', 'say', 'mean', 
    #                          'this', 'think', 'guess', 'just', 'like', 'imagine', 'yes', 'here', 'there', 'so', 'wait','think', 'check']
    
    additional_stop_words = ['so', 'yeah', 'well', 'uh', 'ok', 'now', 'we', 'know', 'that', 'we', 'say', 'mean', 
                             'this', 'think', 'guess', 'just', 'like', 'imagine', 'yes', 'here', 'there', 'let', 
                             'us', 'make', 'wait', 'looks', 'also', 'would', 'one']
    stop_words.update(additional_stop_words)
    
    # Keep these
    words_to_exclude = {'not', 'more', 'less', 'no'}
    stop_words = stop_words - words_to_exclude
    #print('final stopwords', stop_words)
    # Tokenize
    word_tokens = word_tokenize(utterance)
    filtered_utterance = [w for w in word_tokens if w.lower() not in stop_words and w not in string.punctuation]
    

    return " ".join(filtered_utterance)

def load_model(model_dir, verbose=False):
    # Load tokenizer with the local files directly
    tokenizer = AutoTokenizer.from_pretrained(model_dir + '/bert', use_auth_token=False)
    
    # Create an instance of model 
    model = CrossEncoder(is_training=True, long=False, model_name ='bert-base-uncased')  # Change model is something different is being used 
    if verbose:
        print(f"prop extract device: {device}")
    
    #model weights
    model.linear.load_state_dict(torch.load(model_dir + '/linear.chkpt', map_location=device))
    model.model = AutoModel.from_pretrained(model_dir + '/bert')
    #model = torch.nn.DataParallel(model)

    model.to(device)

    if verbose:
        print("new vocab")
        print(tokenizer.vocab['<m>'])  # Check if <m> is in the tokenizer's vocabulary
        print(tokenizer.vocab['</m>']) 
    return model, tokenizer

def process_sentence(sentence, model, tokenizer, verbose=False):
    sentence = remove_stop_words(sentence)
    #inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    common_grounds_dataSet = pd.read_csv('featureModules/prop/data/NormalizedList.csv')
    # common_grounds_dataSet = pd.read_csv('data/NormalizedList.csv')
    common_grounds = list(common_grounds_dataSet['Propositions'])
    
    elements = extract_colors_and_numbers(sentence.lower()) #The list of colors / weights in the transcript
    if 'yellow' in elements["colors"] and '40' in elements["numbers"] and not sentence.strip().endswith('.'):
        sentence += '.'
    if verbose:
        print(elements)
    filtered_common_grounds = []
    filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)]
    if verbose:
        print('common_ground level 1', filtered_common_grounds)
    if not filtered_common_grounds:  # If no match found, try individual color-number pairs
            filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]  #If there is no match where only the mentioned colors and weights are present, get the individual combincations 
    
    if verbose:
        print("length of filtered common grounds:", len(filtered_common_grounds))

    if len(filtered_common_grounds) > 100:
        print(f"WARNING: {len(filtered_common_grounds)} common grounds, processing will likely take a long time")

    if len(filtered_common_grounds) > 137:
        print("Using cosine similarity")
        return get_simple_cosine(sentence, filtered_common_grounds)

    cosine_similarities = get_cosine_similarities(sentence, filtered_common_grounds, model, device, tokenizer)
    top_matches = sorted(zip(filtered_common_grounds, cosine_similarities), key=lambda x: x[1], reverse=True)[:5]
    new_rows = append_matches(top_matches, sentence)

    new_df = pd.DataFrame(new_rows, columns=["transcript", "common_ground"])
    new_df.index.to_list()#the list of indicies in the dict that needs to be tokenized
    
    proposition_map_test = new_df.to_dict(orient='index') #make it into a dict
    proposition_map_test = add_special_tokens(proposition_map_test)    # add the special tokens to transcript and common ground 

    #call tokenize props here.
    
    new_df.to_csv("test_set.csv") #sanity check
    
    
    test_ab, test_ba = tokenize_props(tokenizer, new_df.index.to_list(), proposition_map_test, model.end_id, max_sentence_len=512, truncate=True)    
    
    test_scores_ab, test_scores_ba = predict_with_XE(model, test_ab, test_ba, device, 4,cosine_sim=False)
    test_predictions = (test_scores_ab + test_scores_ba)/2
    new_df["scores"] = test_predictions #Get the raw scores as given by the cross Encoder
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
    if verbose:
        print("test preds")
        print(test_predictions)
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
    if verbose:
        print("new df")
        print(new_df)
    highest_score_row = new_df.loc[new_df['scores'].idxmax()]
    #print(new_df)
    # Extract the 'common_ground' value from this row
    highest_score_common_ground = highest_score_row['common_ground']
    if verbose:
        print("highest score")
        print(highest_score_common_ground)
    return highest_score_common_ground, len(filtered_common_grounds)


#Testing 
# sentence = 'I think Blue is greater than 20'
# model, tokenizer = load_model('/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/XE/googleSandbox/XE_models/model') #changed for testing 
# print(process_sentence(sentence,model,tokenizer))
