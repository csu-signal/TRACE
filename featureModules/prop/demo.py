import pickle
import sys
import torch
#from helper import tokenize, forward_ab, f1_score, accuracy, precision, recall
import pandas as pd
import random
from tqdm import tqdm
import os
from models import CrossEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import re
from transformers import AutoTokenizer
import torch
#from cosine_sim import CrossEncoder_cossim
import torch.nn.functional as F
#from helperMethods import is_proposition_present, normalize_expression, normalize_sub_expression, extract_colors
from demoHelpers import tokenize_props, extract_colors_and_numbers, is_valid_common_ground,\
is_valid_individual_match, predict_with_XE, add_special_tokens
import torch
from transformers import AutoModel, AutoTokenizer
import os




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def load_model(model_dir):
    # Load tokenizer with the local files directly
    tokenizer = AutoTokenizer.from_pretrained(model_dir + '/bert', use_auth_token=False)
    
    # Create an instance of your model with appropriate flags
    model = CrossEncoder(is_training=True, long=False, model_name ='bert-base-uncased')  # Ensure these flags are set as needed for your use case
    model.to(device)
    
    # Load the model weights
    model.linear.load_state_dict(torch.load(model_dir + '/linear.chkpt', map_location=device))
    model.model = AutoModel.from_pretrained(model_dir + '/bert')
    #model = torch.nn.DataParallel(model)

    print(tokenizer.vocab['<m>'])  # Check if <m> is in the tokenizer's vocabulary
    print(tokenizer.vocab['</m>']) 
    return model, tokenizer
def process_sentence(sentence, model, tokenizer):
    #inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    common_grounds_dataSet = pd.read_csv('data/NormalizedList.csv')
    common_grounds = list(common_grounds_dataSet['Propositions'])
    
    elements = extract_colors_and_numbers(sentence.lower()) #The list of colors / weights in the transcript
    filtered_common_grounds = []
    filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)]
    if not filtered_common_grounds:  # If no match found, try individual color-number pairs
            filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]  #If there is no match where only the mentioned colors and weights are present, get the individual combincations 
    cosine_similarities = []
    
    for cg in filtered_common_grounds:
        cg_with_token = "<m>" + " " + cg + " "  + "</m>"
        trans_with_token = "<m>" + " "+ sentence +" " + "</m>"
        theIndividualDict = {
            "transcript": trans_with_token,
            "common_ground": cg_with_token # match[0] is the common ground text
        }
        
        proposition_map = {0: theIndividualDict} 
        proposition_ids = [0]
        tokenizer = tokenizer
        #print(model.end_id)
       
        test_ab, test_ba = tokenize_props(tokenizer,proposition_ids,proposition_map,model.end_id ,max_sentence_len=512, truncate=True)    
        
        
        
        cosine_test_scores_ab, cosine_test_scores_ba = predict_with_XE(model, test_ab, test_ba, device, 4,cosine_sim=True)
        cosine_similarity = (cosine_test_scores_ab + cosine_test_scores_ba) /2
        cosine_similarities.append(cosine_similarity)
    top_matches = sorted(zip(filtered_common_grounds, cosine_similarities), key=lambda x: x[1], reverse=True)[:5]
    new_rows = []
    for match in top_matches:
        new_row = {
            "transcript": sentence,
            "common_ground": match[0]  # match[0] is the common ground text
        }
        new_rows.append(new_row)
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
    print(test_predictions)
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
    print(new_df)
    highest_score_row = new_df.loc[new_df['scores'].idxmax()]

    # Extract the 'common_ground' value from this row
    highest_score_common_ground = highest_score_row['common_ground']
    print(highest_score_common_ground)
    return highest_score_common_ground
model_dir = 'data/chk_42/'
model, tokenizer = load_model(model_dir)


sentence = "I think red is ten"
output = process_sentence(sentence, model, tokenizer)
print(output)