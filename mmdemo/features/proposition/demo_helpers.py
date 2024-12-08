import os
import pickle
import re
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

NORMALIZED_PROP_LIST = str(Path(__file__).parent / "data" / "NormalizedList.csv")
CG_EMBEDDINGS_PKL = str(Path(__file__).parent / "cg_embeddings.pkl")


def add_special_tokens(proposition_map):
    for x, y in proposition_map.items():
        # print(y['common_ground'])
        cg_with_token = "<m>" + " " + y["common_ground"] + " " + "</m>"
        # print(y['transcript'])
        prop_with_token = "<m>" + " " + y["transcript"] + " " + "</m>"
        proposition_map[x]["common_ground"] = cg_with_token
        proposition_map[x]["transcript"] = prop_with_token
    return proposition_map


def tokenize_props(
    tokenizer,
    proposition_ids,
    proposition_map,
    m_end,
    max_sentence_len=1024,
    truncate=True,
):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = "<doc-s>"
    doc_end = "</doc-s>"

    for index in proposition_ids:
        sentence_a = proposition_map[index]["transcript"]
        sentence_b = proposition_map[index]["common_ground"]

        def make_instance(sent_a, sent_b):
            return " ".join(["<g>", doc_start, sent_a, doc_end]), " ".join(
                [doc_start, sent_b, doc_end]
            )

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = (
                input_id[curr_start_index:m_end_index]
                + input_id[m_end_index : m_end_index + (max_sentence_len // 4)]
            )
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (
                max_sentence_len // 2 - len(in_truncated)
            )
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a["input_ids"])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b["input_ids"])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {
            "input_ids": tokenized_ab_,
            "attention_mask": (tokenized_ab_ != tokenizer.pad_token_id),
            "position_ids": positions_ab,
        }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)
    else:
        instances_ab = [" ".join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [" ".join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(
            list(instances_ab), add_special_tokens=False, padding=True
        )

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab["input_ids"])

        tokenized_ab = {
            "input_ids": torch.LongTensor(tokenized_ab["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_ab["attention_mask"]),
            "position_ids": torch.arange(tokenized_ab_input_ids.shape[-1]).expand(
                tokenized_ab_input_ids.shape
            ),
        }

        tokenized_ba = tokenizer(
            list(instances_ba), add_special_tokens=False, padding=True
        )
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba["input_ids"])
        tokenized_ba = {
            "input_ids": torch.LongTensor(tokenized_ba["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_ba["attention_mask"]),
            "position_ids": torch.arange(tokenized_ba_input_ids.shape[-1]).expand(
                tokenized_ba_input_ids.shape
            ),
        }

    return tokenized_ab, tokenized_ba


def get_arg_attention_mask(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    # input_ids.cpu()

    num_inputs = input_ids.shape[0]

    m_start_indicator = input_ids == parallel_model.start_id
    m_end_indicator = input_ids == parallel_model.end_id

    m = m_start_indicator + m_end_indicator

    # non-zero indices are the tokens corresponding to <m> and </m>
    nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1], device=nz_indexes.device)
    q = q.repeat(m.shape[0], 1)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    # attention_mask_g = None
    # attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()

    return attention_mask_g, arg1, arg2


def forward_ab(
    parallel_model, ab_dict, device, indices, lm_only=False, cosine_sim=False
):
    batch_tensor_ab = ab_dict["input_ids"][indices, :].to(device)
    batch_am_ab = ab_dict["attention_mask"][indices, :].to(device)
    batch_posits_ab = ab_dict["position_ids"][indices, :].to(device)
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    if am_g_ab is not None:
        am_g_ab = am_g_ab.to(device)
    arg1_ab = arg1_ab.to(device)
    arg2_ab = arg2_ab.to(device)

    return parallel_model(
        batch_tensor_ab,
        attention_mask=batch_am_ab,
        position_ids=batch_posits_ab,
        global_attention_mask=am_g_ab,
        arg1=arg1_ab,
        arg2=arg2_ab,
        lm_only=lm_only,
        cosine_sim=False,
    )


def predict_with_XE(
    parallel_model, dev_ab, dev_ba, device, batch_size, cosine_sim=False
):
    n = dev_ab["input_ids"].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    description = "Predicting"
    if cosine_sim:
        description = "Getting Cosine"
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_indices = indices[i : i + batch_size]
            scores_ab = forward_ab(
                parallel_model, dev_ab, device, batch_indices, cosine_sim=False
            )
            scores_ba = forward_ab(
                parallel_model, dev_ba, device, batch_indices, cosine_sim=False
            )
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)


number_mapping = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def extract_colors_and_numbers(text):
    colors = ["red", "blue", "green", "yellow", "purple"]
    numbers = ["10", "20", "30", "40", "50"]
    found_elements = {"colors": [], "numbers": []}
    for color in colors:
        if color in text:
            found_elements["colors"].append(color)
    for number in numbers:
        if number in text:
            found_elements["numbers"].append(number)
    return found_elements


def is_valid_common_ground_1(cg, elements):
    cg_colors = re.findall(r"\b(?:red|blue|green|yellow|purple)\b", cg)
    cg_numbers = re.findall(r"\b(?:10|20|30|40|50)\b", cg)
    cg_set = set(cg_colors + cg_numbers)  # Combine and convert to set

    # Combine colors and numbers from the elements dictionary into a set
    # Assume elements['colors'] and elements['numbers'] are provided as lists
    element_colors = elements.get("colors", [])
    element_numbers = [str(num) for num in elements.get("numbers", [])]
    elements_set = set(element_colors + element_numbers)

    return cg_set == elements_set


def is_valid_common_ground_2(cg, elements):
    cg_colors = re.findall(r"\b(?:red|blue|green|yellow|purple)\b", cg)
    cg_numbers = [str(num) for num in re.findall(r"\b(?:10|20|30|40|50)\b", cg)]
    # print(cg_colors, cg_numbers)
    color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
    number_match = not elements["numbers"] or set(cg_numbers) == set(
        elements["numbers"]
    )
    return color_match and number_match


def is_valid_individual_match(cg, elements):
    cg_colors = re.findall(r"\b(?:red|blue|green|yellow|purple)\b", cg)
    cg_numbers = [str(num) for num in re.findall(r"\b(?:10|20|30|40|50)\b", cg)]
    if elements["colors"] == []:
        elements["colors"] = [None]
    if elements["numbers"] == []:
        elements["numbers"] = [None]
    for color in elements["colors"]:
        for number in elements["numbers"]:
            if color in cg_colors or number in cg_numbers:
                return True
    return False


def get_pickle(bert):
    emb_path = CG_EMBEDDINGS_PKL
    if os.path.isfile(emb_path):
        with open(emb_path, "rb") as file:
            embeddings = pickle.load(file)
    else:
        cg_data = pd.read_csv(NORMALIZED_PROP_LIST)
        props = list(cg_data["Propositions"])
        embeddings = get_cg_embeddings(props, bert, {})
        with open(emb_path, "wb") as write:
            pickle.dump(embeddings, write)
    return embeddings


def get_cg_embeddings(filtered_common_grounds, bert, embeddings):
    for cg in tqdm(
        filtered_common_grounds, desc="Generating prop embeddings for cos sim"
    ):
        emb = bert.encode(cg, convert_to_tensor=True)
        embeddings[cg] = emb

    return embeddings


def sentence_fcg_cosine(cg_embedding, sentence_embedding):
    cosine_score = util.cos_sim(sentence_embedding, cg_embedding)
    return cosine_score


def get_sentence_embedding(sentence, bert):
    sentence_embedding = bert.encode(sentence, convert_to_tensor=True)
    return sentence_embedding


def append_matches(top_matches, sentence):
    new_rows = []
    for match in top_matches:
        new_row = {
            "transcript": sentence,
            "common_ground": match[0],  # match[0] is the common ground text
        }
        new_rows.append(new_row)
    return new_rows


def get_simple_cosine(sentence, filtered_common_grounds, bert, embeddings, device):
    cg_cosine_scores = []
    # embeddings = get_cg_embeddings(filtered_common_grounds, bert, embeddings)
    sentence_embedding = get_sentence_embedding(sentence, bert)
    for cg in filtered_common_grounds:
        cosine_score = sentence_fcg_cosine(
            embeddings[cg].to(device), sentence_embedding.to(device)
        ).item()
        cg_cosine_scores.append([sentence, cg, cosine_score])
    df_cosine_scores = pd.DataFrame(
        cg_cosine_scores, columns=["sentence", "common ground", "scores"]
    )
    highest_score_row = df_cosine_scores.loc[df_cosine_scores["scores"].idxmax()]
    highest_score_common_ground = highest_score_row["common ground"]

    return highest_score_common_ground, len(filtered_common_grounds)


def get_cosine_similarities(
    sentence, filtered_common_grounds, model, device, tokenizer
):
    cosine_similarities = []
    for cg in filtered_common_grounds:
        cg_with_token = "<m>" + " " + cg + " " + "</m>"
        trans_with_token = "<m>" + " " + sentence + " " + "</m>"
        theIndividualDict = {
            "transcript": trans_with_token,
            "common_ground": cg_with_token,  # match[0] is the common ground text
        }

        proposition_map = {0: theIndividualDict}
        proposition_ids = [0]
        tokenizer = tokenizer
        # print(model.end_id)

        test_ab, test_ba = tokenize_props(
            tokenizer,
            proposition_ids,
            proposition_map,
            model.end_id,
            max_sentence_len=512,
            truncate=True,
        )

        cosine_test_scores_ab, cosine_test_scores_ba = predict_with_XE(
            model, test_ab, test_ba, device, 4, cosine_sim=True
        )
        cosine_similarity = (cosine_test_scores_ab + cosine_test_scores_ba) / 2
        cosine_similarities.append(cosine_similarity)
    return cosine_similarities
