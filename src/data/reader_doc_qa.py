import numpy as np
import torch
import random
from collections import defaultdict

from data.data_structures_modal import *
from data.data_preparation_modal import make_training_data, make_test_data, make_test_data_from_doc_lst
from data.data_preparation_modal import eng_max_sent_distance, chn_max_sent_distance, \
    eng_e2e_sent_distance, chn_e2e_sent_distance
# from data.tokenization import tokenize_doc_no_overlap, tokenize_doc_with_overlap, tokenize_query_context
from data.tokenization import tokenize_doc_with_overlap, tokenize_query_context

EVENT_START_MARK = 'Event'
EVENT_END_MARK = 'Event'
AUTHOR_TOKEN = 'Author'
NON_CONC_TOKEN = 'No'
ROOT_TOKNE = 'Root'

labeled = True
# modal_edge_label_set = MODAL_EDGE_LABEL_LIST
# temporal_edge_label_set = TEMPORAL_EDGE_LABEL_LIST


class Doc:
    def __init__(self, doc_id, sentences_before_tokenization, sentences, nodes,
                 gold_edges, gold_one_hot=None,
                 sent_subtokens=None, sent_token_id2subtoken_id=None,
                 bio=None, bio_idx=None,
                 pred_edges=None, pred_one_hot=None, pred_nodes=None,
                 doc_subtoken2token=None, doc_token2subtoken=None,
                 child2gold_parent=None,
                 language=None,
                 data_type=None,
                 current_child=None):

        self.doc_id = doc_id
        self.sentences_before_tokenization = sentences_before_tokenization
        self.sentences = sentences  # a list of sentences
        self.nodes = nodes  # a list of nodes

        self.gold_edges = gold_edges
        self.gold_one_hot = gold_one_hot

        self.sent_subtokens = sent_subtokens
        self.sent_token_id2subtoken_id = sent_token_id2subtoken_id

        self.bio = bio
        self.bio_idx = bio_idx

        self.pred_edges = pred_edges
        self.pred_one_hot = pred_one_hot
        self.pred_nodes = pred_nodes

        self.doc_subtoken2token = doc_subtoken2token
        self.doc_token2subtoken = doc_token2subtoken

        self.child2gold_parent = child2gold_parent
        self.language = language
        self.data_type = data_type
        self.current_child = current_child


def get_a_list_of_docs_upper_train(input_file, language):
    #    assert 'test' not in input_file
    gold_data, train_doc_id = make_training_data(input_file, language=language)
    a_list_of_mini_docs = get_a_list_of_docs(
        gold_data, train_doc_id, labeled=labeled, is_training=True, data_type='modal', language=language)
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test(input_file, language):
    test_data, test_doc_id = make_test_data(input_file, language=language)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='modal', language=language)
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_train_tdt_h(input_file, language='eng'):
    #    assert 'test' not in input_file
    gold_data, train_doc_id = make_training_data_tdt_h(input_file)
    a_list_of_mini_docs = get_a_list_of_docs(
        gold_data, train_doc_id, labeled=labeled, is_training=True, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test_tdt_h(input_file, language='eng'):
    test_data, test_doc_id = make_test_data_tdt_h(input_file)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test_from_doc_lst(doc_lst):
    test_data, test_doc_id = make_test_data_from_doc_lst(doc_lst)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_train_temporal(input_file, parent_type='time'):
    assert 'test' not in input_file
    gold_data, train_doc_id = make_training_data_temporal(
        input_file, parent_type=parent_type)
    a_list_of_mini_docs = get_a_list_of_docs(
        gold_data, train_doc_id, labeled=labeled, is_training=True, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test_temporal(input_file, parent_type='time'):
    test_data, test_doc_id = make_test_data_temporal(
        input_file, parent_type=parent_type)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test_temporal_from_doc_lst(doc_lst):
    test_data, test_doc_id = make_test_data_from_doc_lst_temporal(doc_lst)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='temporal')
    return a_list_of_mini_docs


def get_a_list_of_docs(data, doc_id_list, labeled, is_training, data_type, language):
    """
    Generage gold labels for each child node,
    and return a list of Doc.
    """
    if data_type == 'modal':
        edge_label_set = MODAL_EDGE_LABEL_LIST
    else:
        assert "temporal" in data_type
        edge_label_set = TEMPORAL_EDGE_LABEL_LIST

    if language == "chn":
        max_sent_distance = chn_max_sent_distance
    else:
        assert language == "eng"
        max_sent_distance = eng_max_sent_distance

    if not isinstance(max_sent_distance, int):
        # for Chinese data
        max_sent_before, max_sent_after = max_sent_distance
    else:
        max_sent_before = max_sent_distance
        max_sent_after = max_sent_distance

    # # for debugging only
    # if language == "chn":
    #     assert max_sent_before == 1000
    #     assert max_sent_after == 3
    # else:
    #     assert max_sent_before == max_sent_after == 5

    all_doc_list = []
    all_qa_list = []
    for doc_idx, document in enumerate(data):
        doc_id = doc_id_list[doc_idx]
        sentence_list, child_parent_candidates, node_list = document
        # if is_training:
        #     random.shuffle(child_parent_candidates)

        child2parent = {}
        if is_training:
            for child in child_parent_candidates:
                for cand_p in child:
                    p, c, l = cand_p
                    assert c.ID == child[0][1].ID
                    if l is not None and l != 'NO_EDGE':
                        assert l in edge_label_set
                        child2parent[c.ID] = (p, l)
        else:
            for child in child_parent_candidates:
                for cand_p in child:
                    p, c, l = cand_p
                    child2parent[c.ID] = (p, l)

        for child in child_parent_candidates:
            # Do not need to find the parent for conceivers and meta-nodes, which will be the grandparent of events,
            # that is, for each event, we find its parent (conceiver),
            # and its grandparent (conceiver of conceiver)

            if child[0][1].label.startswith('C'):
                continue
            if child[0][1].ID.startswith('-5'):
                continue
            if child[0][1].ID.startswith('-3'):
                continue

            one_doc_example = Doc(doc_id=doc_id,
                                  sentences_before_tokenization=sentence_list,
                                  sentences=sentence_list,
                                  nodes=node_list,
                                  gold_edges=child_parent_candidates,
                                  child2gold_parent=child2parent,
                                  current_child=child[0][1],
                                  language=language,
                                  data_type=data_type)

            question, context, context_sent_idx = get_one_child_qa(
                child, sentence_list, max_sent_before, max_sent_after, EVENT_START_MARK, EVENT_END_MARK)
            flat_context, flat_context_sent_map, token_idx_in_sent_map = get_context_sent_map(
                context, context_sent_idx)

            flat_context.append(AUTHOR_TOKEN)
            context_sent_idx.append(-3)
            flat_context_sent_map.append(-3)
            token_idx_in_sent_map.append(-3)

            flat_context.append(NON_CONC_TOKEN)
            context_sent_idx.append(-5)
            flat_context_sent_map.append(-5)
            token_idx_in_sent_map.append(-5)

            all_qa_list.append([question, flat_context, context_sent_idx, flat_context_sent_map,
                                token_idx_in_sent_map])
            all_doc_list.append(one_doc_example)
    return [all_doc_list, all_qa_list]


def get_one_child_qa(child_node, sentence_list, max_sent_before, max_sent_after,
                     mention_start_mark, mention_end_mark):
    """Generate query and context for a child node"""
    child_snt, child_t_s, child_t_e = [int(t) for t in child_node[0][1].ID.split('_')]

    # Make sure meta-nodes are not included
    assert child_snt >= 0

    query = sentence_list[child_snt][:child_t_s] + [mention_start_mark] + [sentence_list[child_snt][child_t_s]] + \
            [mention_end_mark] + sentence_list[child_snt][child_t_e + 1:]
    # # To avoid error during truncation
    # if len(query) > 130:
    #     query = [mention_start_mark] + [sentence_list[child_snt][child_t_s]] + \
    #             [mention_end_mark] + sentence_list[child_snt][child_t_e + 1:]
    #     query = query[:130]

    sent_before_child_sent = sentence_list[max(child_snt - max_sent_before, 0): child_snt]
    sent_after_child_sent = sentence_list[child_snt + 1:(child_snt + max_sent_after + 1)]

    context = sent_before_child_sent + [sentence_list[child_snt]] + sent_after_child_sent

    sent_before_ids = [child_snt - i for i in range(1, len(sent_before_child_sent) + 1)]
    context_sent_idx = sent_before_ids[::-1] + [child_snt] + [child_snt + i for i in
                                                              range(1, len(sent_after_child_sent) + 1)]
    return query, context, context_sent_idx


def get_context_sent_map(context, context_sent_idx):
    flat_context = []
    sent_map = []
    token_idx_in_sent_map = []
    # Context is a list of sentences, each sentence is a list of tokens;
    # context_sent_idx contains the sent_id in original document for each sent in context.
    for s_idx, sent in enumerate(context):
        token_sent_idx = context_sent_idx[s_idx]
        for t_id, token in enumerate(sent):
            flat_context.append(token)
            sent_map.append(token_sent_idx)
            token_idx_in_sent_map.append(t_id)
    return flat_context, sent_map, token_idx_in_sent_map


def tokenize_docs_and_update_answers(qa_list, original_examples, max_seq_len, tokenizer, is_training):
    querys = [doc[0] for doc in qa_list]
    contexts = [doc[1] for doc in qa_list]

    # doc_stride_ratio = 1/2
    # doc_stride = 64
    # is_split_into_words = True
    tokenized_examples = tokenize_query_context(querys, contexts, tokenizer, max_seq_len,
                                                is_split_into_words=True, doc_stride=64)

    updated_tokenized_examples = []
    for i, tokenized_qa in enumerate(tokenized_examples):
        orig_example_id = tokenized_qa["id_before_tokenization"]
        orig_example = original_examples[orig_example_id]

        doc_id = orig_example.doc_id
        cur_child = orig_example.current_child.ID
        orig_sent_list = orig_example.sentences_before_tokenization

        question, flat_context, context_sent_idx, flat_context_sent_map, token_idx_in_sent_map = qa_list[orig_example_id]

        tokenized_qa["doc_id"] = doc_id
        tokenized_qa["current_child"] = cur_child
        tokenized_qa["questions"] = question
        tokenized_qa["contexts"] = flat_context
        tokenized_qa["context_sent_idx"] = context_sent_idx
        tokenized_qa["flat_context_sent_map"] = flat_context_sent_map
        tokenized_qa["token_idx_in_sent_map"] = token_idx_in_sent_map

        offset_before_tokenization = sum([len(s) for s in orig_sent_list[:context_sent_idx[0]]])

        if is_training:
            gold_parent, gold_rel = orig_example.child2gold_parent[cur_child]
            if gold_parent.ID in orig_example.child2gold_parent and gold_parent.ID != "-3_-3_-3":
                grand_parent, grand_rel = orig_example.child2gold_parent[gold_parent.ID]
            else:
                grand_parent, grand_rel = None, None

            if gold_parent.snt_index_in_doc >= 0:
                parent_start_in_context = gold_parent.start_word_index_in_doc - offset_before_tokenization
                parent_end_in_context = gold_parent.end_word_index_in_doc - offset_before_tokenization
                assert " ".join(flat_context[parent_start_in_context:parent_end_in_context + 1]) == gold_parent.words
            elif gold_parent.snt_index_in_doc == -3:
                parent_start_in_context, parent_end_in_context = len(flat_context) - 2, len(flat_context) - 2
                assert " ".join(flat_context[parent_start_in_context:parent_end_in_context + 1]) == AUTHOR_TOKEN
            else:
                assert gold_parent.snt_index_in_doc == -5
                parent_start_in_context, parent_end_in_context = len(flat_context) - 1, len(flat_context) - 1
                assert " ".join(flat_context[parent_start_in_context:parent_end_in_context + 1]) == NON_CONC_TOKEN

            if parent_start_in_context in tokenized_qa["token2subtoken_maps"] and \
                    parent_end_in_context in tokenized_qa["token2subtoken_maps"]:
                parent_subtoken_start_in_context = tokenized_qa["token2subtoken_maps"][parent_start_in_context][0]
                parent_subtoken_end_in_context = tokenized_qa["token2subtoken_maps"][parent_end_in_context][-1]
            else:
                parent_subtoken_start_in_context, parent_subtoken_end_in_context = None, None

            if grand_parent:
                if grand_parent.snt_index_in_doc >= 0:
                    grand_parent_start_in_context = grand_parent.start_word_index_in_doc - offset_before_tokenization
                    grand_parent_end_in_context = grand_parent.end_word_index_in_doc - offset_before_tokenization
                    assert " ".join(flat_context[
                                    grand_parent_start_in_context:grand_parent_end_in_context + 1]) == grand_parent.words
                elif grand_parent.snt_index_in_doc == -3:
                    grand_parent_start_in_context, grand_parent_end_in_context = len(flat_context) - 2, len(
                        flat_context) - 2
                    assert " ".join(
                        flat_context[grand_parent_start_in_context:grand_parent_end_in_context + 1]) == AUTHOR_TOKEN
                else:
                    assert grand_parent.snt_index_in_doc == -5
                    grand_parent_start_in_context, grand_parent_end_in_context = len(flat_context) - 1, len(
                        flat_context) - 1
                    assert " ".join(
                        flat_context[grand_parent_start_in_context:grand_parent_end_in_context + 1]) == NON_CONC_TOKEN

                if grand_parent_start_in_context in tokenized_qa["token2subtoken_maps"] and \
                        grand_parent_end_in_context in tokenized_qa["token2subtoken_maps"]:
                    grand_parent_subtoken_start_in_context = \
                    tokenized_qa["token2subtoken_maps"][grand_parent_start_in_context][0]
                    grand_parent_subtoken_end_in_context = \
                    tokenized_qa["token2subtoken_maps"][grand_parent_end_in_context][-1]
                else:
                    grand_parent_subtoken_start_in_context, grand_parent_subtoken_end_in_context = None, None
            else:
                grand_parent_start_in_context, grand_parent_end_in_context = None, None
                grand_parent_subtoken_start_in_context, grand_parent_subtoken_end_in_context = None, None

            tokenized_qa["answer_in_context"] = (parent_start_in_context, parent_end_in_context)
            tokenized_qa["grand_answer_in_context"] = (grand_parent_start_in_context, grand_parent_end_in_context)
            tokenized_qa["subtoken_answer_in_context"] = (
                parent_subtoken_start_in_context, parent_subtoken_end_in_context)
            tokenized_qa["grand_subtoken_answer_in_context"] = (
                grand_parent_subtoken_start_in_context, grand_parent_subtoken_end_in_context)

        updated_tokenized_examples.append(tokenized_qa)

    return updated_tokenized_examples


def get_bio_for_qa(original_examples, tokenized_examples, max_seq_len):
    tokenized_examples_with_bio = []

    for i, tokenized_qa in enumerate(tokenized_examples):
        orig_example_id = tokenized_qa["id_before_tokenization"]
        orig_example = original_examples[orig_example_id]

        cur_child = orig_example.current_child.ID
        gold_parent, gold_rel = orig_example.child2gold_parent[cur_child]

        if gold_parent.ID in orig_example.child2gold_parent and gold_parent.ID != "-3_-3_-3":
            grand_parent, grand_rel = orig_example.child2gold_parent[gold_parent.ID]
        else:
            grand_parent, grand_rel = None, None

        valid_context_bio = ['o'] * len(tokenized_qa["tokens"])
        valid_context_rel = ['NA'] * len(tokenized_qa["tokens"])

        parent_start, parent_end = tokenized_qa["subtoken_answer_in_context"]
        grand_parent_start, grand_parent_end = tokenized_qa["grand_subtoken_answer_in_context"]

        for t_id, token in enumerate(valid_context_bio):
            if parent_start is not None and t_id == parent_start:
                valid_context_bio[t_id] = 'b_c'
                valid_context_rel[t_id] = gold_rel
            if parent_start is not None and t_id > parent_start and parent_end is not None and t_id <= parent_end:
                valid_context_bio[t_id] = 'i_c'
                valid_context_rel[t_id] = gold_rel
            if grand_parent_start is not None and t_id == grand_parent_start:
                valid_context_bio[t_id] = 'b_g'
                valid_context_rel[t_id] = grand_rel
            if grand_parent_end is not None and t_id > grand_parent_start and \
                    grand_parent_end is not None and t_id <= grand_parent_end:
                valid_context_bio[t_id] = 'i_g'
                valid_context_rel[t_id] = grand_rel

        contxt_start, context_end = tokenized_qa["context_start_end"]
        query_bio = ['o'] * contxt_start
        query_rel = ['NA'] * contxt_start
        padding_bio = ['o'] * (max_seq_len - context_end)
        padding_rel = ['NA'] * (max_seq_len - context_end)

        assert len(query_bio) + len(valid_context_bio) + len(padding_bio) == len(tokenized_qa["input_ids"])
        assert len(query_rel) + len(valid_context_rel) + len(padding_rel) == len(tokenized_qa["input_ids"])

        input_bio = query_bio + valid_context_bio + padding_bio
        input_rel = query_rel + valid_context_rel + padding_rel
        tokenized_qa['input_bio'] = input_bio
        tokenized_qa['input_bio_gold'] = [PARSING_BIO2id[t] for t in input_bio]
        tokenized_qa['input_rel'] = input_rel
        tokenized_qa['input_rel_gold'] = [MODAL_EDGE_LABEL_LIST_SPAN_label2id[t] for t in input_rel]

        tokenized_examples_with_bio.append(tokenized_qa)

    return tokenized_examples_with_bio


def read_data(args, tokenizer, input_file, is_training, data_type, language):
    # if 'test' in input_file or 'author' in input_file:
    #     assert is_training is False

    if isinstance(input_file, list):
        decode_type = 'doc_list'
    else:
        assert isinstance(input_file, str)
        decode_type = 'txt_file'

    if is_training:
        if data_type == 'modal':
            a_list_of_docs = get_a_list_of_docs_upper_train(input_file, language)
        else:
            parent_type = data_type.split('_')[-1]
            assert parent_type in ['time', 'event', 'tdt']
            if parent_type == 'tdt':
                a_list_of_docs = get_a_list_of_docs_upper_train_tdt_h(input_file)
            else:
                a_list_of_docs = get_a_list_of_docs_upper_train_temporal(input_file, parent_type=parent_type)
    else:
        if data_type == 'modal':
            if decode_type == 'doc_list':
                # Assume the input_file is a list
                a_list_of_docs = get_a_list_of_docs_upper_test_from_doc_lst(input_file)
            else:
                a_list_of_docs = get_a_list_of_docs_upper_test(input_file, language)
        else:
            parent_type = data_type.split('_')[-1]
            assert parent_type in ['time', 'event', 'tdt']
            if decode_type == 'doc_list':
                a_list_of_docs = get_a_list_of_docs_upper_test_temporal_from_doc_lst(input_file)
            else:
                if parent_type == 'tdt':
                    a_list_of_docs = get_a_list_of_docs_upper_test_tdt_h(input_file)
                else:
                    a_list_of_docs = get_a_list_of_docs_upper_test_temporal(
                        input_file, parent_type=parent_type)

    # [all_doc_list, all_qa_list]
    # Each item in doc_list and qa_list is essentially a child node
    doc_list, qa_list = a_list_of_docs
    assert len(doc_list) == len(qa_list)
    tokenized_qa_list = tokenize_docs_and_update_answers(qa_list, doc_list,
                                                         args.max_seq_length, tokenizer=tokenizer,
                                                         is_training=is_training)
    if is_training:
        tokenized_qa_list_wt_bio = get_bio_for_qa(doc_list, tokenized_qa_list, args.max_seq_length)
    else:
        tokenized_qa_list_wt_bio = tokenized_qa_list
    return doc_list, tokenized_qa_list_wt_bio


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _convert_list_of_dict2dict_of_list(list_of_example_dict):
    examples_dict_of_list = defaultdict(list)
    keys = list_of_example_dict[0].keys()
    for exa in list_of_example_dict:
        for k in keys:
            examples_dict_of_list[k].append(exa[k])
    return examples_dict_of_list


def convert_to_batch(doc_example_input, tokenized_example_lst, batch_size, max_seq_len, device):
    """
    Get input batch.
    """
    tokenized_doc_lst = []
    for doc_idx, tokenized_doc_exm in enumerate(tokenized_example_lst):
        orig_example_id = tokenized_doc_exm["id_before_tokenization"]
        orig_example = doc_example_input[orig_example_id]
        assert tokenized_doc_exm["doc_id"] == orig_example.doc_id

        tokenized_doc_exm["doc_example"] = orig_example
        tokenized_doc_lst.append(tokenized_doc_exm)

    batchs = list(_chunks(tokenized_doc_lst, batch_size))

    udpated_batchs = []
    for b_id, batch in enumerate(batchs):
        one_batch = _convert_list_of_dict2dict_of_list(batch)

        for k, v in one_batch.items():
            if k == "input_ids":
                one_batch[k] = torch.LongTensor(v).to(device)
            elif k == "attention_mask":
                one_batch[k] = torch.FloatTensor(v).to(device)
            elif k in ["input_bio_gold", "input_rel_gold"]:
                one_batch[k] = torch.LongTensor(v).to(device)
            else:
                one_batch[k] = v

        if b_id != len(batchs) - 1:
            assert one_batch["input_ids"].shape[0] == batch_size
        assert one_batch["input_ids"].shape[-1] == max_seq_len
        assert one_batch["input_ids"].shape == one_batch["attention_mask"].shape
        if len(one_batch["input_bio_gold"]) != 0:
            assert one_batch["input_ids"].shape == one_batch["input_bio_gold"].shape == \
                                                one_batch["input_rel_gold"].shape
        udpated_batchs.append(one_batch)
    return udpated_batchs


def tag_paths_to_spans(path):
    """Convert predicted tag paths to a list of spans.
    """
    mentions = []
    cur_mention = None
    sec_mentions = []
    sec_cur_mention = None
    for j, tag in enumerate(path):
        tag = tag.upper()
        #print(tag, path)
        assert tag[0] in ['B', 'I', 'O']
        if tag == 'O':
            prefix = tag = 'O'
        else:
            prefix, tag = tag.split('_', 1)

        if prefix == 'B':
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = [j, j, tag]
        elif prefix == 'I':
            if cur_mention is not None:
                if cur_mention[-1] == tag:
                    cur_mention[1] = j
                else:
                    mentions.append(cur_mention)
                    cur_mention = None
            else:
                if sec_cur_mention is not None:
                    if sec_cur_mention[-1] == tag:
                        sec_cur_mention[1] = j
                    else:
                        sec_mentions.append(sec_cur_mention)
                        sec_cur_mention = [j, j, tag]
                else:
                    sec_cur_mention = [j, j, tag]
        else:
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = None
            if sec_cur_mention:
                sec_mentions.append(sec_cur_mention)
            sec_cur_mention = None

    if cur_mention:
        assert cur_mention[-1] in ['C', 'T', 'E', 'G']
        mentions.append(cur_mention)
    if sec_cur_mention:
        assert sec_cur_mention[-1] in ['C', 'T', 'E', 'G']
        sec_mentions.append(sec_cur_mention)

    if len(mentions) > 0:
        parents = [[s, e] for s, e, l in mentions if l == 'C']
        grand_p = [[s, e] for s, e, l in mentions if l == 'G']
    else:
        parents = []
        grand_p = []
    if len(sec_mentions) > 0:
        sec_parents = [[s, e] for s, e, l in sec_mentions if l == 'C']
        sec_grand_p = [[s, e] for s, e, l in sec_mentions if l == 'G']
    else:
        sec_parents = []
        sec_grand_p = []
    assert len(parents) + len(grand_p) == len(mentions)
    assert len(sec_parents) + len(sec_grand_p) == len(sec_mentions)
    return parents, grand_p, sec_parents, sec_grand_p

