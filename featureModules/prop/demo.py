import torch
import pandas as pd
from featureModules.prop.models import CrossEncoder
from transformers import AutoTokenizer
from featureModules.prop.demoHelpers import tokenize_props, extract_colors_and_numbers, is_valid_common_ground, \
is_valid_individual_match, predict_with_XE, add_special_tokens, get_embeddings, sentence_fcg_cosine
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

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
    #inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    common_grounds_dataSet = pd.read_csv('featureModules/prop/data/NormalizedList.csv')
    common_grounds = list(common_grounds_dataSet['Propositions'])
    
    elements = extract_colors_and_numbers(sentence.lower()) #The list of colors / weights in the transcript
    if verbose:
        print(elements)
    filtered_common_grounds = []
    filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)]
    if verbose:
        print('common_ground', filtered_common_grounds)
    if not filtered_common_grounds:  # If no match found, try individual color-number pairs
            filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]  #If there is no match where only the mentioned colors and weights are present, get the individual combincations 
    cosine_similarities = []
    
    if verbose:
        print("length of filtered common grounds:", len(filtered_common_grounds))

    if len(filtered_common_grounds) > 100:
        print(f"WARNING: {len(filtered_common_grounds)} common grounds, processing will likely take a long time")

    if len(filtered_common_grounds) > 137:
        print("Using cosine similaroty")
        model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        cg_cosine_scores = []
        for cg in filtered_common_grounds:
            cosine_score = sentence_fcg_cosine(cg, sentence, model).item()
            print(f'Cosine Score is {cosine_score}')
            cg_cosine_scores.append([sentence, cg, cosine_score])
        df_cosine_scores = pd.DataFrame(cg_cosine_scores, columns = ['sentence', 'common ground', 'cosine similarity'])
        highest_score_row = df_cosine_scores.loc[df_cosine_scores['cosine similarity'].idxmax()]
        highest_score_common_ground = highest_score_row['common ground']

        return highest_score_common_ground, len(filtered_common_grounds)

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
