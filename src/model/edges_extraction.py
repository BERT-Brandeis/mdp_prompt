import torch
import random
import numpy as np
import scipy.sparse as sp
from data.data_structures_modal import *
from data.data_preparation_modal import generate_e_conc_from_bio_tag
from data.data_preparation_modal import generate_e_conc_from_bio_tag, get_word_index_in_doc
from data.data_preparation_modal import choose_candidates_padded as choose_candidates_padded_modal
from data.data_preparation_modal import eng_max_sent_distance, chn_max_sent_distance
from data.reader_doc import get_one_hot_for_one_edge
from data.adj import get_gcn_nodes, get_adj, convert_3dsparse_to_4dsparse, sparse_mxs_to_torch_sparse_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_and_sort_event_conc(stage1_tags, bio2label_map, original_example, pred_nodes_ratio, is_training):
    if len(stage1_tags) != 0:
        stage1_tags = stage1_tags.detach().cpu().tolist()

    extracted_tags = [bio2label_map[t] for t in stage1_tags]
    events, concs = generate_e_conc_from_bio_tag(extracted_tags, original_example.data_type)

    snt_ids = []
    snt_token_ids = []
    for s_id, snt in enumerate(original_example.sentences_before_tokenization):
        t = 0
        for _ in snt:
            snt_ids.append(s_id)
            snt_token_ids.append(t)
            t += 1
    snt_ids_map = {i: snt_id for i, snt_id in enumerate(snt_ids)}
    snt_token_ids_map = {i: snt_token_id for i, snt_token_id in enumerate(snt_token_ids)}

    events = [(e[0], e[-1]) for e in events]
    events = [('_'.join([str(snt_ids_map[p[0]]), str(snt_token_ids_map[p[0]]),
                         str(max(snt_token_ids_map[p[-1]], snt_token_ids_map[p[0]]))]),
               'Event') for p in events]

    concs = [(conc[0], conc[-1]) for conc in concs]
    if original_example.data_type == 'modal':
        concs = [('_'.join([str(snt_ids_map[p[0]]), str(snt_token_ids_map[p[0]]),
                            str(max(snt_token_ids_map[p[-1]], snt_token_ids_map[p[0]]))]), 'Conceiver')
                                  for p in concs]
        events = [(e, l) for (e, l) in events if not (e, 'Conceiver') in concs]
    else:
        concs = [('_'.join([str(snt_ids_map[p[0]]), str(snt_token_ids_map[p[0]]), str(snt_token_ids_map[p[-1]])]), 'Timex')
                                  for p in concs]
        events = [(e, l) for (e, l) in events if not (e, 'Timex') in concs]

    def sort_edge_by_node_id_in_doc(edge):
        node, _ = edge
        node_snt, node_token_s, node_token_e = [int(i) for i in node.split('_')]
        node_idx_in_doc = get_word_index_in_doc(
            original_example.sentences_before_tokenization, node_snt, node_token_s)
        return node_idx_in_doc
    one_doc_nodes = events + concs
    one_doc_nodes.sort(key=sort_edge_by_node_id_in_doc)

    if len(one_doc_nodes) > 0:
        if original_example.data_type == 'modal':
            if not one_doc_nodes[0][0] == '-3_-3_-3':
                one_doc_nodes = [['-3_-3_-3', 'Conceiver']] + one_doc_nodes
        else:
            if not one_doc_nodes[0][0].startswith('-7_-7'):
                one_doc_nodes = [['-7_-7_-7', 'Timex']] + one_doc_nodes

    if is_training:
        gold_nodes = {nd.ID: nd.label for nd in original_example.nodes if nd.snt_index_in_doc >= 0}
        assert pred_nodes_ratio > 0
        if pred_nodes_ratio < 1.:
            num_pred_nodes_including = int(len(one_doc_nodes) * pred_nodes_ratio)
        else:
            num_pred_nodes_including = len(one_doc_nodes)
        pred_nodes_including = one_doc_nodes[:num_pred_nodes_including]

        tmp = [k[0] for k in pred_nodes_including]

        for k, v in gold_nodes.items():
            if len(pred_nodes_including) >= len(one_doc_nodes):
                break
            if k not in tmp:
                pred_nodes_including.append((k, v))
                tmp.append(k)
        assert len(pred_nodes_including) <= len(one_doc_nodes)
        one_doc_nodes = pred_nodes_including

    one_doc_nodes.sort(key=sort_edge_by_node_id_in_doc)
    return one_doc_nodes


def generate_modal_data_edges(one_doc_nodes, original_example, is_training):
    node_list = []
    node_list_id = []
    snt_list = original_example.sentences_before_tokenization
    for i, node in enumerate(one_doc_nodes):
        child, c_label = node
        c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]

        if c_end < c_start:
            print("not well formed nodes: ", child)
            c_end = c_start

        if c_snt >= 0:
            c_words = ' '.join(original_example.sentences_before_tokenization[c_snt][c_start:c_end + 1])
        else:
            assert c_snt == -3
            c_words = AUTHOR_word
            # if c_snt == -3:
            #     c_words = AUTHOR_word
        c_node = Node(snt_index_in_doc=c_snt,
                      start_word_index_in_snt=c_start,
                      end_word_index_in_snt=c_end,
                      node_index_in_doc=i,
                      start_word_index_in_doc=get_word_index_in_doc(snt_list, c_snt, c_start),
                      end_word_index_in_doc=get_word_index_in_doc(snt_list, c_snt, c_end),
                      words=c_words,
                      label=c_label)
        node_list.append(c_node)
        node_list_id.append(c_node.ID)
        # if gold_p in node_id2node:
        #     gold_p = node_id2node[gold_p]

    if original_example.language == "chn":
        max_sent_distance = chn_max_sent_distance
    else:
        assert original_example.language == "eng"
        max_sent_distance = eng_max_sent_distance

    test_instance_list = []
    root_node = get_root_node()
    padding_node = get_padding_node()
    null_conceiver_node = get_null_conceiver_node()

    for c_node in node_list:
        if is_training:
            if c_node.ID in original_example.child2gold_parent:
                gold_p, gold_label = original_example.child2gold_parent[c_node.ID]
                if gold_p not in node_list_id:
                    continue
            else:
                continue
            instance = choose_candidates_padded_modal(
                node_list, root_node, padding_node, null_conceiver_node, c_node, gold_p, gold_label, max_sent_distance)
        else:
            instance = choose_candidates_padded_modal(
                node_list, root_node, padding_node, null_conceiver_node, c_node, None, None, max_sent_distance)
        test_instance_list.append(instance)
    return test_instance_list, node_list


def generate_temporal_data_edges(one_doc_nodes, original_example, data_type, is_training):
    parent_type = data_type.split('_')[-1]
    assert parent_type in ['time', 'event']
    node_list = []
    node_id2node = {}
    snt_list = original_example.sentences_before_tokenization
    for i, node in enumerate(one_doc_nodes):
        child, c_label = node
        c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]
        if c_snt >= 0:
            c_words = ' '.join(original_example.sentences_before_tokenization[c_snt][c_start:c_end + 1])
        else:
            if c_snt == -7:
                c_words = DCT_word
        c_node = Node(c_snt, c_start, c_end, i,
                      get_word_index_in_doc(snt_list, c_snt, c_start),
                      get_word_index_in_doc(snt_list, c_snt, c_end),
                      c_words, c_label)
        node_id2node[c_node.ID] = c_node
        node_list.append(c_node)

    test_instance_list = []
    root_node = get_root_node()
    padding_node = get_padding_node()

    for c_node in node_list:
        if parent_type == 'time':
            instance = choose_candidates_timex(
                node_list, root_node, padding_node, c_node, None, None)
        else:
            instance = choose_candidates_event(
                node_list, root_node, padding_node, c_node, None, None)
        test_instance_list.append(instance)

    return test_instance_list

def generate_modal_pred_one_hot(a_list_of_child, edge_label_set):
    one_hot = []
    for child in a_list_of_child:
        one_hot_for_one_child = []
        for one_edge in child:
            p, c, l = one_edge
            one_hot_for_one_edge = get_one_hot_for_one_edge(l, edge_label_set, labeled=True)
            one_hot_for_one_child.append(one_hot_for_one_edge)
        one_hot.append(one_hot_for_one_child)
    return one_hot


def generate_edges(stage1_tags, bio2label_map, original_example, edge_label_set, pred_nodes_ratio, is_training):
    if original_example.data_type == 'modal':
        one_doc_nodes = extract_and_sort_event_conc(
            stage1_tags, bio2label_map, original_example, pred_nodes_ratio, is_training)
        pred_edges, pred_nodes = generate_modal_data_edges(one_doc_nodes, original_example, is_training=is_training)
        if is_training:
            random.shuffle(pred_edges)
            pred_one_hot = generate_modal_pred_one_hot(pred_edges, edge_label_set)
            updated_pred_edges = []
            updated_pred_one_hot = []
            for i, child in enumerate(pred_one_hot):
                has_parent = False
                for j, candidate in enumerate(child):
                    if 1 in candidate:
                        has_parent = True
                if has_parent:
                    updated_pred_edges.append(pred_edges[i])
                    updated_pred_one_hot.append(child)
            pred_edges = updated_pred_edges
            pred_one_hot = updated_pred_one_hot
        else:
            pred_one_hot = None
    else:
        pass

    original_example.pred_edges = pred_edges
    original_example.pred_one_hot = pred_one_hot
    original_example.pred_nodes = pred_nodes
    return original_example


def update_gcn_nodes(original_example):
    gcn_nodes, gcn_node_ids = get_gcn_nodes(mention_node_lst=original_example.pred_nodes,
                                            doc_sent_lst=original_example.sentences_before_tokenization)
    rgcn_adjacency = get_adj(gcn_nodes, mention_nodes_lst=original_example.pred_nodes,
                             return_sparse_tensor=False)
    sent_len_map = [len(s) for s in original_example.sentences_before_tokenization]
    return gcn_nodes, gcn_node_ids, rgcn_adjacency, sent_len_map


def get_regularized_adj(stage1_scores, threshold, nodes, gcn_node_id_map, orig_adj):
    mm, ss, ms = [0, 1, 2]
    weights = np.full((orig_adj.shape[-1], orig_adj.shape[-1]), 1.0)
    # Compute averaged scores for each extracted mention
    for i, nd in enumerate(nodes):
        if nd.snt_index_in_doc < 0:
            continue
        nd_id_in_gcn_matrix = gcn_node_id_map[nd.ID]
        nd_start, nd_end = nd.start_word_index_in_doc, nd.end_word_index_in_doc
        nd_score = stage1_scores[nd_start:nd_end+1]

        # num_token in this node, e.g. torch.Size([2])
        tag_score, tag_idx = nd_score.max(dim=-1)
        nd_score = torch.mean(tag_score).item()

        weights[nd_id_in_gcn_matrix, :] = nd_score
        weights[:, nd_id_in_gcn_matrix] = nd_score

    orig_adj[mm] = orig_adj[mm] * weights
    orig_adj[ms] = orig_adj[mm] * weights

    num_edge_type = orig_adj.shape[0]
    orig_adj = sparse_mxs_to_torch_sparse_tensor(
            [sp.coo_matrix(orig_adj[i]) for i in range(num_edge_type)])
    return orig_adj
