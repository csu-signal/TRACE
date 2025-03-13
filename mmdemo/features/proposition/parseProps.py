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
    sentence = re.sub(r"(\b\w+)'s\s+(\w+)", r"\1 is \2", sentence)
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
