import re
import string
import pandas as pd
def normalize_expression(expr):
   
    sub_expressions = expr.split(',')
    normalized_sub_expressions = [normalize_sub_expression(sub.strip()) for sub in sub_expressions]
  
    if all('=' in sub and (sub.strip().split('=')[0].strip().isalpha() and sub.strip().split('=')[1].strip().isdigit()) for sub in normalized_sub_expressions):
        normalized_sub_expressions.sort()
    return ', '.join(normalized_sub_expressions)

def normalize_sub_expression(sub_expr):
 
    match = re.search(r'([=!<>]+)', sub_expr)
    if match:
        operator = match.group(1)
        parts = re.split(r'([=!<>]+)', sub_expr, 1)
        left_side = parts[0].strip()
        right_side = parts[2].strip()

       
        if '+' in right_side:
            right_side_components = re.findall(r'\w+', right_side)
            right_side_sorted = ' + '.join(sorted(right_side_components))
            return f"{left_side} {operator} {right_side_sorted}"
        elif operator in ['=', '!=']:
 
            if not right_side.isdigit() and left_side > right_side:  
                return f"{right_side} {operator} {left_side}"
            else:
                return sub_expr
        else:
      
            return sub_expr
    else:

        return sub_expr


def extract_propositions_ltr_v2(sentence, additional_stopwords=None):

    blocks = {"red", "blue", "green", "purple", "yellow"}
    weights = {"10", "20", "30", "40", "50"}

  
    relation_map = {
        ">": {"heavier", "more", "heavy", "bigger", "greater", "above"},
        "<": {"lighter", "less", "small", "below", "under"},
        "=": {"equals", "equal", "same", "is"}, 
        "!=": {"different", "unequal", "isn't", "aren't"}
    }

    invert_map = {
        ">": "<",
        "<": ">",
        "=": "!=",
        "!=": "="
    }

    stop_words = {
        "oh", "i", "think", "than", "a", "an", "the", "that", "this", "it",
        "to", "of", "and", "or", "on", "in", "at", "with", "for", "but", "so",
        "when", "we", "look", "its"
    }
    if additional_stopwords:
        stop_words.update(additional_stopwords)


    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    raw_tokens = sentence.lower().split()
    tokens = [w for w in raw_tokens if w not in stop_words]


    def get_relation_symbol(token):
        for symbol, synonyms in relation_map.items():
            if token in synonyms:
                return symbol
        return None

    
    recognized_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "not":
            if i + 1 < len(tokens) and get_relation_symbol(tokens[i+1]) is not None:
                rel = get_relation_symbol(tokens[i+1])
                inverted = invert_map.get(rel, rel)
                recognized_tokens.append(inverted)
                i += 2
            else:
                recognized_tokens.append("not")
                i += 1
        else:
            rel = get_relation_symbol(token)
            if rel is not None:
                recognized_tokens.append(rel)
            else:
                recognized_tokens.append(token)
            i += 1

    filtered = []
    for idx, token in enumerate(recognized_tokens):
        if token in blocks or token in weights or token in relation_map or token == "not":
            filtered.append((token, idx))
    
   
    num_explicit = sum(1 for token, _ in filtered if token in relation_map)

   
    propositions = []
    left_group = []       # list of entity tokens 
    current_relation = None  # explicit relation token 
    right_group = []      # list of entity tokens after the relation
    gap_tokens = []       # stray "not" tokens
   
    last_right_pos = None

    # Iterate over the filtered tokens.
    for idx, (token, pos) in enumerate(filtered):
        if token in relation_map:
            # Token is an explicit relation.
            if not left_group:
                continue  # no left-hand side yet
            else:
                if current_relation is not None and right_group:
                    # A previous explicit relation is pending and we have some right-hand entities.
                    # If more than one explicit relation appears in the sentence, finalize the current clause
                    # using only the first entity of right_group.
                    if num_explicit > 1:
                        relation_used = (invert_map.get(current_relation, current_relation)
                                         if any(x == "not" for x in gap_tokens)
                                         else current_relation)
                        left_side = " + ".join(left_group)
                        right_side = right_group[0]
                        if left_side != right_side:
                            propositions.append(f"{left_side} {relation_used} {right_side}")
                        # Start a new clause: the remainder of right_group (if any) becomes the new left_group.
                        left_group = right_group[1:] if len(right_group) > 1 else []
                        right_group = []
                        gap_tokens = []
                    # Otherwise (only one explicit relation in the sentence) we leave right_group intact.
                # In any case, update the current relation.
                current_relation = token
                gap_tokens = []
        elif token == "not":
            gap_tokens.append(token)
        elif token in blocks or token in weights:
         
            if current_relation is None:
               
                if left_group and left_group[-1] == token:
                    continue
                left_group.append(token)
            else:
                # An explicit relation is pending; decide whether to accumulate this entity
                # into the right_group or if it signals the start of a new clause.
                # We use a lookahead: if the next token (if any) is an explicit relation,
                # then this entity likely belongs to a new clause.
                lookahead_relation = False
                if idx + 1 < len(filtered):
                    next_token, _ = filtered[idx + 1]
                    if next_token in relation_map:
                        lookahead_relation = True
                if lookahead_relation and right_group:
                    # Finalize the current clause using only the first entity from right_group.
                    relation_used = (invert_map.get(current_relation, current_relation)
                                     if any(x == "not" for x in gap_tokens)
                                     else current_relation)
                    left_side = " + ".join(left_group)
                    right_side = right_group[0]
                    if left_side != right_side:
                        propositions.append(f"{left_side} {relation_used} {right_side}")
                    # Start a new clause with the current token.
                    left_group = [token]
                    current_relation = None
                    right_group = []
                    gap_tokens = []
                else:
                    # Otherwise, accumulate the entity into right_group.
                    right_group.append(token)
                    last_right_pos = pos

    if current_relation is not None:
        if num_explicit == 1:
            # For a single explicit relation, combine all accumulated right-hand entities.
            if right_group:
                relation_used = (invert_map.get(current_relation, current_relation)
                                 if any(x == "not" for x in gap_tokens)
                                 else current_relation)
                left_side = " + ".join(left_group)
                right_side = " + ".join(right_group)
                if left_side != right_side:
                    propositions.append(f"{left_side} {relation_used} {right_side}")
            elif len(left_group) >= 2:
                relation_used = (invert_map.get(current_relation, current_relation)
                                 if any(x == "not" for x in gap_tokens)
                                 else current_relation)
                left_side = " + ".join(left_group[:-1])
                right_side = left_group[-1]
                if left_side != right_side:
                    propositions.append(f"{left_side} {relation_used} {right_side}")
        else:
            
            if right_group:
                relation_used = (invert_map.get(current_relation, current_relation)
                                 if any(x == "not" for x in gap_tokens)
                                 else current_relation)
                left_side = " + ".join(left_group)
                right_side = right_group[0]
                if left_side != right_side:
                    propositions.append(f"{left_side} {relation_used} {right_side}")
                right_group = right_group[1:]
       
            if len(right_group) >= 2:
                left_side = " + ".join(right_group[:-1])
                right_side = right_group[-1]
                propositions.append(f"{left_side} = {right_side}")
    elif not current_relation and len(left_group) >= 2:
     
        implicit_relation = "!=" if any(x == "not" for x in gap_tokens) else "="
        if len(left_group) > 2:
            left_side = " + ".join(left_group[:-1])
        else:
            left_side = left_group[0]
        right_side = left_group[-1]
        if left_side != right_side:
            propositions.append(f"{left_side} {implicit_relation} {right_side}")
    my_string = ",".join(map(str, propositions))
    mainProp = normalize_expression(my_string)
    print(mainProp)
    
   # print('valid props',valid_props)
    return normalize_expression(mainProp)

# if __name__ == "__main__":
#     # Example usage
    
#     s1 = "when we look at red and green its not heavier than blue and the yellow is 50"
#     props1 = extract_propositions_ltr_v2(s1)
#     # print("Sentence:", s1)
#     print("Propositions:", props1)
#     # # Expect => [{"proposition": "red < blue", ...}]


#     s2 = "I think red is not equal to its like absurd blue "
#     props2 = extract_propositions_ltr_v2(s2)
#     print("\nSentence:", s2)
#     print("Propositions:", props2)
#     # Expect => [{"proposition": "red != blue", ...}]

#     s3 = "yellow green are heavier than red"
#     props3 = extract_propositions_ltr_v2(s3)
#     print("\nSentence:", s3)
#     print("Propositions:", props3)
#     # Expect => [{"proposition": "yellow > red", ...}, {"proposition": "green > red", ...}]

#     s1 = "when we look at red its not heavier than blue"
#     props1 = extract_propositions_ltr_v2(s1)
#     print("Sentence:", s1)
#     print("Propositions:", props1)
#     # Expect => ["red < blue"]

#     s2 = "I think red is not equal to blue"
#     props2 = extract_propositions_ltr_v2(s2)
#     print("\nSentence:", s2)
#     print("Propositions:", props2)
#     # Expect => ["red != blue"]

#     s3 = "yellow green are heavier than red"
#     props3 = extract_propositions_ltr_v2(s3)
#     print("\nSentence:", s3)
#     print("Propositions:", props3)
#     # Expect => ["yellow > red", "green > red"]
    
  
#     s1 = "10 or 20 seems a bit high for red  and the red is equal to the blue"
#     print("Sentence:", s1)
#     props1 = extract_propositions_ltr_v2(s1)
#     print("Propositions:", props1)
    
#     s1 = "I do not think that red is greater than blue but "
#     print("Sentence:", s1)
#     props1 = extract_propositions_ltr_v2(s1)
#     print("Propositions:", props1)
#     # Expect: ["yellow > red", "green > red"]

#     s2 = "and purple block is equal 30 right and red block is equal 10 so purple block, red block would be 40 so yellow block's more than 40 and less than 60 so yellow block's gotta be 50"
#     props2 = extract_propositions_ltr_v2(s2)
#     print("\nSentence:", s2)
#     print("Propositions:", props2)
#     # Expect: ["red < 10"]
#     s2 = "idk purple or green equals 30 pr 40"
#     props2 = extract_propositions_ltr_v2(s2)
#     print("\nSentence:", s2)
#     print("Propositions:", props2)
#     # Expect: ["red < 10"]
#     s1 = "red 10"
#     props1 = extract_propositions_ltr_v2(s1)
#     print("Sentence:", s1)
#     print("Propositions:", props1)
#     # Expect => [{"proposition": "red = 10", ...}]

#     s2 = "when we look at red its not heavier than blue but probably not"
#     props2 = extract_propositions_ltr_v2(s2)
#     print("\nSentence:", s2)
#     print("Propositions:", props2)
#     # Expect => [{"proposition": "red < blue", ...}]

#     s3 = "I think red is not equal to blue"
#     props3 = extract_propositions_ltr_v2(s3)
#     print("\nSentence:", s3)
#     print("Propositions:", props3)
#     # Expect => [{"proposition": "red != blue", ...}]
#     s3 = "yellow course not  10"
#     props3 = extract_propositions_ltr_v2(s3)
#     print("\nSentence:", s3)
#     print("Propositions:", props3)
#     # Expect => [{"proposition": "red != blue", ...}]
    
#     listOfProps = ["blue is not 10",
#               "I think yellow is definetely not 40.", 
#               " Red 10",
#               "It seems like blue might be about the same as the red",
#               " Green looks like about 20.",
#               "So, yellow block is noticeably heavier than the purple one.",
#               "So red is a 10 and green is a 20 right there ",
#               "I would say purple about 30",
#               "Wait, lets make sure purple is not also 20 ",
#               "so Purple is more than 20.",
#               "Purple has to be 30",
#               "I would guess the yellow would be  equal to 40 ",
#               "Did we say purple was 30 ",
#               "Yeah, purple is 30, green is 20, and blue and red are both 10.",
#               "the yellow is not 40",
#               "Yeah, green plus purple is yellow ",
#               "purple is more than 20",
#               "purple is not 20",
#               "Yeah, we did try yellow as 40 because purple block is 30",
#               "So, purple is 30, and blue is 10? ",
#               "So, yellow is at least closest to 50",
#               "Green looks like about 20",
#               "Blue is a 10 and green is a 20",
#               "Lets double check that purple is also not 20",
#               "I would say that purple is about 30",
#               "Blue is the same as the red", 
#               "Yellow is noticebly heavier than purple",
#               "Yellow is definitely not 40, it's too heavy.",
#               "red and blue are not the same ",
#               "yellow block's for sure not 30",
#               "green is red and blue",
#               "yellow yellow is greater than 10"
#               ]
#     for item in listOfProps:
#         print(item, extract_propositions_ltr_v2(item))