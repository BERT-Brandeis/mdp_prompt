import numpy as np
import torch
import random

from data.data_structures_modal import *
from data.data_preparation_modal import make_training_data, make_test_data, make_test_data_from_doc_lst
#from data_preparation_temporal import make_training_data as make_training_data_temporal
#from data_preparation_temporal import make_test_data as make_test_data_temporal
from data.tokenization import tokenize_doc_no_overlap, tokenize_doc_with_overlap
from data.adj import get_gcn_nodes, get_adj, convert_3dsparse_to_4dsparse


modal_edge_label_set = MODAL_EDGE_LABEL_LIST
temporal_edge_label_set = TEMPORAL_EDGE_LABEL_LIST
labeled = True
max_edges_per_doc = 16
max_sent_per_doc = 1000


class Doc:
    def __init__(self, doc_id, sentences_before_tokenization, sentences, nodes,
                 gold_edges, gold_one_hot=None,
                 sent_subtokens=None, sent_token_id2subtoken_id=None,
                 bio=None, bio_idx=None,
                 pred_edges=None, pred_one_hot=None, pred_nodes=None,
                 doc_subtoken2token=None, doc_token2subtoken=None,
                 child2gold_parent=None,
                 language=None,
                 data_type=None):

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


def get_a_list_of_docs_upper_train(input_file, language):
    #    assert 'test' not in input_file
    gold_data, train_doc_id = make_training_data(input_file, language=language)
    a_list_of_mini_docs = get_a_list_of_docs(
        gold_data, train_doc_id, labeled=labeled, is_training=True, data_type='modal')
    return a_list_of_mini_docs


def get_a_list_of_docs_upper_test(input_file, language):
    test_data, test_doc_id = make_test_data(input_file, language=language)
    a_list_of_mini_docs = get_a_list_of_docs(
        test_data, test_doc_id, labeled=labeled, is_training=False, data_type='modal')
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


def get_a_list_of_docs(data, doc_id_list, labeled, is_training, data_type):
    """
    Generage gold labels for each child node,
    and return a list of Doc.
    """
    if data_type == 'modal':
        edge_label_set = modal_edge_label_set
    else:
        assert "temporal" in data_type
        edge_label_set = temporal_edge_label_set

    all_doc_list = []
    for doc_idx, document in enumerate(data):
        doc_id = doc_id_list[doc_idx]
        sentence_list, child_parent_candidates, node_list = document
        # if len(sentence_list) > max_sent_per_doc:
        #     continue  # skip long doc to fit gpu
        if is_training:
            random.shuffle(child_parent_candidates)

        # To train a doc with too many edges in multiple steps, in order to fit gpu
        if len(child_parent_candidates) > max_edges_per_doc:
           a_list_of_child_nodes_chunked = list(_chunks(child_parent_candidates, max_edges_per_doc))
        else:
           a_list_of_child_nodes_chunked = [child_parent_candidates]

        child2parent = {}
        for a_list_of_child in a_list_of_child_nodes_chunked:
            for child in a_list_of_child:
                for cand_p in child:
                    p, c, l = cand_p
                    assert c.ID == child[0][1].ID
                    if l is not None and l != 'NO_EDGE':
                        assert l in edge_label_set
                        child2parent[c.ID] = (p.ID, l)

        for a_list_of_child in a_list_of_child_nodes_chunked:
            edges = []
            one_hot = []
            for child in a_list_of_child:

                one_hot_for_one_child = []
                for one_edge in child:
                    p, c, l = one_edge
                    if is_training:
                        one_hot_for_one_edge = get_one_hot_for_one_edge(l, edge_label_set, labeled)
                    else:
                        one_hot_for_one_edge = get_fake_one_hot_for_one_edge(edge_label_set, labeled)
                    one_hot_for_one_child.append(one_hot_for_one_edge)

                edges.append(child)
                one_hot.append(one_hot_for_one_child)

            one_doc_example = Doc(doc_id=doc_id,
                                  sentences_before_tokenization=sentence_list,
                                  sentences=sentence_list,
                                  nodes=node_list,
                                  gold_edges=edges,
                                  gold_one_hot=one_hot,
                                  child2gold_parent=child2parent)
            all_doc_list.append(one_doc_example)
    return all_doc_list


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_one_hot_for_one_edge(label_for_candp, edge_label_set, labeled):
    candidate_one_hot = []
    if labeled:
        for label in edge_label_set:
            if label == label_for_candp:
                candidate_one_hot.append(1)
            else:
                candidate_one_hot.append(0)
    elif not labeled:
        if label_for_candp != 'NO_EDGE':
            candidate_one_hot.append(1)
        else:
            candidate_one_hot.append(0)
    return np.array(candidate_one_hot)


def get_fake_one_hot_for_one_edge(edge_label_set, labeled):
    candidate_one_hot = []
    if labeled:
        for _ in edge_label_set:
            candidate_one_hot.append(0)
    elif not labeled:
        candidate_one_hot.append(0)
    assert 1 not in candidate_one_hot
    return candidate_one_hot

def _flatten(lst):
    """Convert a list of list to list."""
    flatten_lst = []
    for upper_lst in lst:
        for item in upper_lst:
            flatten_lst.append(item)
    return flatten_lst


def get_event_conc_bio_input(nodes, doc, is_training, event=True, conc=True):
    """
    :param nodes: a list of nodes
    :param doc: a list of sentences
    """
    input_bio_list = [['o'] * len(snt) for snt in doc]
    input_bio_gold_label = [[EVENT_CONC_BIO2id['o']] * len(snt) for snt in doc]

    if not is_training:
        input_bio_list = _flatten(input_bio_list)
        input_bio_gold_label = _flatten(input_bio_gold_label)
        return input_bio_list, input_bio_gold_label

    start_label_map = {"E": "b_e", "C": "b_c", "T": "b_t"}
    middle_end_label_map = {"E": "i_e", "C": "i_c", "T": "i_t"}

    # Ignore meta nodes
    nodes = [nd for nd in nodes if nd.snt_index_in_doc >= 0]
    for nd in nodes:
        assert nd.label in ["Event", "Conceiver", "Timex"]

    assert event or conc
    if not conc:
        nodes = [nd for nd in nodes if nd.label[0] != 'C']
    if not event:
        nodes = [nd for nd in nodes if nd.label[0] != 'E']

    for node in nodes:
        nd_label = node.label
        assert nd_label[0] in ["E", "C", "T"]

        c_snt, s_token_idx, e_token_idx = [int(t) for t in node.ID.split('_')]
        assert c_snt >= 0 # make sure meta nodes are not included

        for s_idx, sent in enumerate(doc):
            if c_snt == s_idx:
                for idx, tag in enumerate(sent):
                    if idx == s_token_idx:
                        input_bio_list[s_idx][idx] = start_label_map[nd_label[0]]
                        input_bio_gold_label[s_idx][idx] = EVENT_CONC_BIO2id[start_label_map[nd_label[0]]]
                    if s_token_idx < idx < e_token_idx:
                        input_bio_list[s_idx][idx] = middle_end_label_map[nd_label[0]]
                        input_bio_gold_label[s_idx][idx] = EVENT_CONC_BIO2id[middle_end_label_map[nd_label[0]]]
                    if idx == e_token_idx:
                        if s_token_idx == e_token_idx:
                            continue
                        input_bio_list[s_idx][idx] = middle_end_label_map[nd_label[0]]
                        input_bio_gold_label[s_idx][idx] = EVENT_CONC_BIO2id[middle_end_label_map[nd_label[0]]]
            else:
                continue
    input_bio_list = _flatten(input_bio_list)
    input_bio_gold_label = _flatten(input_bio_gold_label)
    return input_bio_list, input_bio_gold_label


def prepare_one_doc(doc, max_seq_len, tokenizer, encoding_method, char_based):
    if encoding_method == "overlap":
        tokenize_fnc = tokenize_doc_with_overlap
        doc_stride_ratio = 1/2
    else:
        assert encoding_method == "no_overlap"
        tokenize_fnc = tokenize_doc_no_overlap
        doc_stride_ratio = False

    if char_based:
        is_split_into_words = False
    else:
        is_split_into_words = True

    doc_input_id_chunks, left_pads, right_pads, token_ratio, doc_attention_mask_chunks, \
    doc_token2subtoken_map, doc_subtoken2token, whole_doc_input_ids = \
        tokenize_fnc(doc, tokenizer, max_seq_len, is_split_into_words, doc_stride_ratio)

    doc_token_len = [v[-1] - v[0] + 1 for k, v in doc_token2subtoken_map.items() if k is not None]
    token_len_tmp = sum([len(s) for s in doc.sentences_before_tokenization])
    assert len(doc_token_len) == token_len_tmp
    assert len(doc_token_len) - 1 == doc_subtoken2token[-2]

    gcn_nodes, gcn_node_ids = get_gcn_nodes(mention_node_lst=doc.nodes, doc_sent_lst=doc.sentences_before_tokenization)
    rgcn_adjacency = get_adj(gcn_nodes, mention_nodes_lst=doc.nodes)

    sent_len_map = [len(s) for s in doc.sentences_before_tokenization]
    gcn_input = {"gcn_nodes": gcn_nodes,
                 "gcn_node_id_map": gcn_node_ids,
                 "adj": rgcn_adjacency,
                 "sent_len_map": sent_len_map}

    pretrained_lm_input = {"input_ids": [torch.LongTensor(input_id)
                                         for input_id in doc_input_id_chunks],
                           "attention_mask": [torch.FloatTensor(attention_mask)
                                              for attention_mask in doc_attention_mask_chunks],
                           "token_lens": doc_token_len,
                           "num_token_pad_to_left": left_pads,
                           "num_token_pad_to_right": right_pads,
                           "token_ratio": token_ratio,
                           "whole_doc_input_ids": torch.LongTensor(whole_doc_input_ids)}

    one_doc_example_input = (doc_token2subtoken_map, doc_subtoken2token)

    return one_doc_example_input, pretrained_lm_input, gcn_input


def read_data(args, tokenizer, input_file, is_training, data_type, language):
    if 'test' in input_file or 'author' in input_file:
        assert is_training is False

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
    # if language == "chn":
    #     char_based = True
    # else:
    #     char_based = False
    # char_based = False

    updated_docs = []
    lm_inputs, gcn_inputs = [], []

    for one_doc in a_list_of_docs:
        one_doc_example_input, pretrained_lm_input, gcn_input = prepare_one_doc(
            doc=one_doc, max_seq_len=args.max_seq_length, tokenizer=tokenizer,
            encoding_method=args.encoding_method, char_based=False)

        token2subtoken_map, subtoken2token = one_doc_example_input
        input_bio_list, input_bio_gold_label = get_event_conc_bio_input(
            one_doc.nodes, one_doc.sentences_before_tokenization, is_training=is_training,
            event=args.extract_event, conc=args.extract_conc)
        one_doc_example = Doc(doc_id=one_doc.doc_id,
                              sentences_before_tokenization=one_doc.sentences_before_tokenization,
                              sentences=one_doc.sentences_before_tokenization,
                              nodes=one_doc.nodes,
                              gold_edges=one_doc.gold_edges,
                              gold_one_hot=one_doc.gold_one_hot,
                              doc_subtoken2token=subtoken2token,
                              doc_token2subtoken=token2subtoken_map,
                              bio=input_bio_list,
                              bio_idx=torch.LongTensor(input_bio_gold_label),
                              data_type=data_type,
                              language=language,
                              child2gold_parent=one_doc.child2gold_parent
                              )
        updated_docs.append(one_doc_example)
        lm_inputs.append(pretrained_lm_input)
        gcn_inputs.append(gcn_input)
    return updated_docs, lm_inputs, gcn_inputs


def convert_to_batch(one_doc_example_input, pretrained_lm_input, gcn_input, device):
    """
    Get input batch.
    """
    batch = []
    for doc_idx, doc_exm in enumerate(one_doc_example_input):
        # print("doc_idx", doc_idx, doc_exm.doc_id)
        doc_lm_input = pretrained_lm_input[doc_idx]
        for k, v in doc_lm_input.items():
            if k in ["input_ids", "attention_mask"]:
                doc_lm_input[k] = [ts.to(device) for ts in v]
            elif k == "whole_doc_input_ids":
                doc_lm_input[k] = v.to(device)
            elif k == "token_ratio":
                doc_lm_input[k] = torch.tensor(v, dtype=torch.float32).to(device)
            else:
                doc_lm_input[k] = v

        doc_gcn_input = gcn_input[doc_idx]
        for k, v in doc_gcn_input.items():
            if k == "adj":
                v = [v]
                doc_gcn_input[k] = convert_3dsparse_to_4dsparse(v).to(device)
            elif k == "gcn_node_id_map":
                doc_gcn_input[k] = v
            else:
                doc_gcn_input[k] = torch.LongTensor(v).to(device)

        one_batch = {"doc_example": doc_exm,
                     "pretrained_lm_input": doc_lm_input,
                     "gcn_input": doc_gcn_input}
        batch.append(one_batch)
    return batch
