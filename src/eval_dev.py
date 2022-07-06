import torch
import codecs
import collections
from collections import defaultdict
from collections import Counter

from eval import eval_all, eval_stage1, get_labeled_tups, get_event_tups, get_te_tups
from data.data_structures_modal import MODAL_EDGE_LABEL_LIST, TEMPORAL_EDGE_LABEL_LIST, id2EVENT_CONC_BIO
from data.data_structures_modal import id2PARSING_BIO, MODAL_EDGE_LABEL_LIST_SPAN_id2label
from data.data_preparation_modal import generate_e_conc_from_bio_tag, get_word_index_in_doc, \
    tag_paths_to_spans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def readin_tuples(lines, from_file=False):
    if from_file:
        lines = codecs.open(lines, 'r', 'utf-8').readlines()
    else:
        lines = lines.split('\n')
    edge_tuples = []
    mode = None
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            edge = line.strip().split('\t')
            # print(filename, edge)
            assert len(edge) in [2, 4]
            if len(edge) == 4:
                child, child_label, parent, link_label = edge
                edge_tuples[-1].append((child, parent, link_label))
            else:
                child, child_label = edge
                edge_tuples[-1].append((child, child_label))
    return edge_tuples


def readin_stage1_tuples(lines, from_file=False):
    if from_file:
        lines = codecs.open(lines, 'r', 'utf-8').readlines()
    else:
        lines = lines.split('\n')

    edge_tuples = []
    mode = None
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            edge = line.strip().split('\t')
            assert len(edge) in [2, 4]
            if len(edge) == 4:
                child, child_label, parent, link_label = edge
            else:
                child, child_label = edge
            # Ignore meta nodes when evaluating stage1
            if int(child.split('_')[0]) < 0:
                continue
            edge_tuples[-1].append((child, child_label))
    return edge_tuples


def write_stage1_parsed(predict_nodes, data_type):
    string = ''
    previous_doc_id = 'Null'
    added_nodes = []
    for doc_example, doc_nodes in predict_nodes.items():
        doc_id = doc_example.doc_id
        events, concs = [], []
        for lst in doc_nodes:
            e, c = lst
            events.extend(e)
            concs.extend(c)
        if doc_id != previous_doc_id:
            # Start a new doc
            if previous_doc_id != 'Null':
                string += '\n\n'
            added_nodes = []
            sent_list = '\n'.join([' '.join(sent) for sent in doc_example.sentences_before_tokenization])
            string += 'filename:' + doc_id + ':SNT_LIST' + '\n' + sent_list + '\nEDGE_LIST\n'
        previous_doc_id = doc_id
        for conc in concs:
            conc_snt, conc_s, conc_e = conc
            conc = str(conc_snt) + '_' + str(conc_s) + '_' + str(conc_e)
            if conc not in added_nodes:
                added_nodes.append(conc)
                if data_type == 'modal':
                    string += '\t'.join([conc, 'Conceiver'])
                else:
                    string += '\t'.join([conc, 'Timex'])
                string += '\n'
        for e in events:
            e_snt, e_s, e_e = e
            e = str(e_snt) + '_' + str(e_s) + '_' + str(e_e)
            if e not in added_nodes:
                added_nodes.append(e)
                string += '\t'.join([e, 'Event'])
                string += '\n'
    return string


def get_dev_stage1_f(dev_parsed, gold_file):
    gold_tuples = readin_stage1_tuples(gold_file, from_file=True)
    auto_tuples = readin_stage1_tuples(dev_parsed)
    # e_macro_f, e_micro_f, te_macro_f, te_micro_f
    e_macro_f, e_micro_f, conc_macro_f, conc_micro_f = eval_stage1(
        gold_tuples, auto_tuples, get_event_tups, get_te_tups)
    return e_macro_f, e_micro_f, conc_macro_f, conc_micro_f


def write_stage2_parsed(list_of_edges):
    string = ''
    previous_doc_id = 'Null'
    one_doc_edge = []
    for doc_example, doc_edges in list_of_edges.items():
        doc_id = doc_example.doc_id
        if not doc_id == previous_doc_id:
            # Start a new doc
            if previous_doc_id != 'Null':
                string += '\n\n'
            one_doc_edge = []
            sent_list = '\n'.join([' '.join(sent) for sent in doc_example.sentences_before_tokenization])
            string += 'filename:' + doc_id + ':SNT_LIST' + '\n' + sent_list + '\n' + 'EDGE_LIST\n'
        previous_doc_id = doc_id
        for edge in doc_edges:
            one_doc_edge.append(edge[0])
            string += '\t'.join(edge)
            string += '\n'
        # Check no duplicate
        assert len(one_doc_edge) == len(set(one_doc_edge))
    return string


def get_parent_type(p, id2label):
    if p in id2label:
        return id2label[p]
    else:
        return "Conceiver"


def write_span_parsed(input_parent_list, input_label_list):
    string = ''

    for doc_id, exp in input_parent_list.items():
        label_exp = input_label_list[doc_id]
        # Start a new doc
        one_doc_edge = []
        one_doc_to_add = []
        one_doc_edge.append('-3_-3_-3')
        sent_list = '\n'.join([' '.join(sent) for sent in exp[0][0].sentences_before_tokenization])
        string += 'filename:' + doc_id + \
                  ':SNT_LIST' + '\n' + sent_list + '\n' + 'EDGE_LIST\n'
        string += '\t'.join(['-3_-3_-3', 'Conceiver', '-1_-1_-1', 'Depend-on'])
        string += '\n'

        child_this_doc = {c.ID: c.label for c in exp[0][0].nodes}
        pred_conc2conc = {}
        pred_conc = set()

        for i, lst in enumerate(exp):
            doc_examples, parent_lst, grand_p_list, _, _ = lst
            _, parent_label_lst, grand_p_label_lst, _, _ = label_exp[i]

            grand_p, grand_p_label = None, None
            parent, parent_label = None, None
            # If there are more than one parent extracted, take the first one
            if len(parent_lst) >= 1:
                parent = parent_lst[0]
                parent = '_'.join([str(t) for t in parent])
                for l_i, l in enumerate(parent_label_lst):
                    if l != 'NA':
                        parent_label = l
                        break
                if not parent_label:
                    parent_label = 'NA'
            if len(grand_p_list) >= 1:
                grand_p = grand_p_list[0]
                grand_p = '_'.join([str(t) for t in grand_p])
                for l1_i, l1 in enumerate(grand_p_label_lst):
                    if l1 != 'NA':
                        grand_p_label = l1
                        break
                if not grand_p_label:
                    grand_p_label = 'NA'
            if parent:
                if parent == doc_examples.current_child.ID:
                    continue
                # parent = parent2valid_parent[parent]
                one_edge = [doc_examples.current_child.ID, doc_examples.current_child.label, parent, parent_label]
                if not one_edge[0] in one_doc_edge:
                    one_doc_edge.append(one_edge[0])
                    string += '\t'.join(one_edge)
                    string += '\n'
                pred_conc.add(parent)
            if grand_p:
                if grand_p == parent:
                    continue
                if parent in ['-3_-3_-3', '-5_-5_-5']:
                    continue
                    # grand_p = parent2valid_parent[grand_p]
                if parent:
                    one_edge = [parent, get_parent_type(parent, child_this_doc), grand_p, grand_p_label]
                    if not one_edge[0] in one_doc_edge:
                        one_doc_edge.append(one_edge[0])
                        string += '\t'.join(one_edge)
                        string += '\n'
                pred_conc2conc[parent] = (get_parent_type(parent, child_this_doc), grand_p, grand_p_label)
                pred_conc.add(grand_p)

        for to_add in pred_conc:
            if to_add in one_doc_edge or to_add in ['-3_-3_-3', '-5_-5_-5']:
                continue
            if to_add in pred_conc2conc:
                conc_type, conc_of_conc, conc_label = pred_conc2conc[to_add]
                one_edge = [to_add, conc_type, conc_of_conc, conc_label]
                one_doc_edge.append(to_add)
                string += '\t'.join(one_edge)
                string += '\n'
            else:
                if to_add in child_this_doc:
                    string += '\t'.join([to_add, child_this_doc[to_add], '-3_-3_-3', 'pos'])
                else:
                    string += '\t'.join([to_add, 'Conceiver', '-3_-3_-3', 'pos'])
            string += '\n'
        string += '\n\n'
    return string


def write_span_parsed_v2(input_parent_list, input_label_list, input_valid_parent=None):
    string = ''
    for doc_id, exp in input_parent_list.items():
        label_exp = input_label_list[doc_id]
        if input_valid_parent is not None:
            parent2valid_parent = input_valid_parent[doc_id]
        # Start a new doc
        one_doc_edge = []

        sent_list = '\n'.join([' '.join(sent) for sent in exp[0][0].sentences_before_tokenization])
        string += 'filename:' + doc_id + \
                  ':SNT_LIST' + '\n' + sent_list + '\n' + 'EDGE_LIST\n'
        string += '\t'.join(['-3_-3_-3', 'Conceiver', '-1_-1_-1', 'Depend-on'])
        string += '\n'
        one_doc_edge.append('-3_-3_-3')

        child_this_doc = {c.ID: c.label for c in exp[0][0].nodes}

        pred_child2parent = {}
        pred_child2grand_p = {}
        pred_conc2conc = {}
        pred_conc = set()

        for i, lst in enumerate(exp):
            doc_examples, parent_lst, grand_p_lst, sec_parent_lst, sec_grand_p_lst = lst
            _, parent_label_lst, grand_p_label_lst, sec_parent_label_lst, sec_grand_p_label_lst = label_exp[i]

            parent, parent_label = None, None
            grand_p, grand_p_label = None, None
            if len(parent_lst) == 0:
                if len(sec_parent_lst) > 0:
                    parent_lst = sec_parent_lst
                    parent_label_lst = sec_parent_label_lst
            # If there are more than one parent extracted, take the first one
            if len(parent_lst) >= 1:
                parent = parent_lst[0]
                parent = '_'.join([str(t) for t in parent])
                if input_valid_parent is not None:
                    if parent in parent2valid_parent:
                        parent = parent2valid_parent[parent]
                for l_i, l in enumerate(parent_label_lst):
                    if l != 'NA':
                        parent_label = l
                        break
                if not parent_label:
                    parent_label = 'NA'

            if len(grand_p_lst) == 0:
                if len(sec_grand_p_lst) > 0:
                    grand_p_lst = sec_grand_p_lst
                    grand_p_label_lst = sec_grand_p_label_lst
            if len(grand_p_lst) >= 1:
                grand_p = grand_p_lst[0]
                grand_p = '_'.join([str(t) for t in grand_p])
                if input_valid_parent is not None:
                    if grand_p in parent2valid_parent:
                        grand_p = parent2valid_parent[grand_p]
                for l1_i, l1 in enumerate(grand_p_label_lst):
                    if l1 != 'NA':
                        grand_p_label = l1
                        break
                if not grand_p_label:
                    grand_p_label = 'NA'

            if parent and parent != doc_examples.current_child.ID:
                pred_child2parent[doc_examples.current_child.ID] = (
                    doc_examples.current_child.label, parent, parent_label)
                pred_conc.add(parent)

            if grand_p and grand_p != doc_examples.current_child.ID:
                grand_p_type = get_parent_type(grand_p, child_this_doc)
                pred_child2grand_p[doc_examples.current_child.ID] = (grand_p_type, grand_p, grand_p_label)
                if parent and parent != doc_examples.current_child.ID and parent != grand_p:
                    pred_conc2conc[parent] = (get_parent_type(parent, child_this_doc), grand_p, grand_p_label)
                pred_conc.add(grand_p)

        for child, values in pred_child2parent.items():
            child_label, parent, parent_label = values
            one_edge = [child, child_label, parent, parent_label]
            one_doc_edge.append(child)
            string += '\t'.join(one_edge)
            string += '\n'

        for child, g_values in pred_child2grand_p.items():
            grand_p_type, grand_p, grand_p_label = g_values
            if child in pred_child2parent:
                child_parent = pred_child2parent[child][-2]
                child_parent_type = get_parent_type(child_parent, child_this_doc)
                if child_parent not in one_doc_edge:
                    if child_parent == grand_p or child_parent in ["-3_-3_-3", "-5_-5_-5"]:
                        continue
                    one_edge = [child_parent, child_parent_type, grand_p, grand_p_label]
                    one_doc_edge.append(child_parent)
                    string += '\t'.join(one_edge)
                    string += '\n'
            else:
                # Assume this grand p is p
                if grand_p != child and child not in ["-3_-3_-3", "-5_-5_-5"]:
                    child_parent = grand_p
                    one_edge = [child, doc_examples.current_child.label, child_parent, grand_p_label]
                    one_doc_edge.append(child)
                    string += '\t'.join(one_edge)
                    string += '\n'
        for conc, c_values in pred_conc2conc.items():
            conc_type, conc_of_conc, conc_label = c_values
            if conc not in one_doc_edge:
                one_edge = [conc, conc_type, conc_of_conc, conc_label]
                one_doc_edge.append(child)
                string += '\t'.join(one_edge)
                string += '\n'
        # If this conceiver doesn't have a conceiver, attach it to Author to form a well-defined tree
        for conc in pred_conc:
            if conc not in ["-3_-3_-3", "-5_-5_-5"] and conc not in one_doc_edge:
                conc_type = get_parent_type(conc, child_this_doc)
                one_edge = [conc, conc_type, '-3_-3_-3', 'pos']
                one_doc_edge.append(conc)
                string += '\t'.join(one_edge)
                string += '\n'
        string += '\n\n'
    return string


def get_dev_stage2_f(dev_parsed, gold_file):
    gold_tuples = readin_tuples(gold_file, from_file=True)
    auto_tuples = readin_tuples(dev_parsed)
    macro_f, micro_f = eval_all(gold_tuples, auto_tuples, get_labeled_tups, labeled=True)
    return macro_f, micro_f


def parse_pipeline_stage2(model, eval_features, data_type):
    if data_type == 'modal':
        EDGE_LABEL_LIST = MODAL_EDGE_LABEL_LIST
    else:
        assert "temporal" in data_type
        EDGE_LABEL_LIST = TEMPORAL_EDGE_LABEL_LIST

    model.eval()
    parsed_child = []
    list_of_predict_edges = collections.defaultdict(list)
    for step, eval_feature in enumerate(eval_features):
        original_example = eval_feature["doc_example"]
        with torch.no_grad():
            label_scores = model(one_doc_example=eval_feature["doc_example"],
                         lm_input=eval_feature["pretrained_lm_input"],
                         gcn_input=eval_feature["gcn_input"],
                         is_training=False,
                         use_pred_edges=False)
        if len(label_scores) == 0:
            continue
        _, predict_rel = label_scores.max(dim=-1)
        num_instances = predict_rel.tolist()
        for idx, p in enumerate(num_instances):
            p_parent_idx = int(predict_rel[idx] / len(EDGE_LABEL_LIST))
            p_label = predict_rel[idx] % len(EDGE_LABEL_LIST)
            p_parent = original_example.gold_edges[idx][p_parent_idx][0]
            c = original_example.gold_edges[idx][p_parent_idx][1]
            if (original_example.doc_id, c.ID) not in parsed_child:
                parsed_child.append((original_example.doc_id, c.ID))
            else:
                continue
            p_label = EDGE_LABEL_LIST[p_label]
            list_of_predict_edges[original_example].append((c.ID, c.label, p_parent.ID, p_label))

    for k, v in list_of_predict_edges.items():
        assert len(v) == len(set(v))
    assert len(parsed_child) == len(set(parsed_child))

    string = write_stage2_parsed(list_of_predict_edges)
    return string


def get_doc_stage1_nodes(tags, doc_example, id2bio, scores=None):
    extracted_tags = [id2bio[t] for t in tags]
    events_doc, concs_doc = generate_e_conc_from_bio_tag(extracted_tags, doc_example.data_type)
    ## For debugging only
    #events_doc1, concs_doc1 = tag_paths_to_spans(extracted_tags)
    #assert events_doc == events_doc1
    #assert concs_doc == concs_doc1

    snt_ids = []
    snt_token_ids = []
    for s_id, snt in enumerate(doc_example.sentences_before_tokenization):
        t = 0
        for _ in snt:
            snt_ids.append(s_id)
            snt_token_ids.append(t)
            t += 1
    snt_ids_map = {i: snt_id for i, snt_id in enumerate(snt_ids)}
    snt_token_ids_map = {i: snt_token_id for i, snt_token_id in enumerate(snt_token_ids)}

    events = [(e[0], e[-1]) for e in events_doc]
    events = [[snt_ids_map[p[0]], snt_token_ids_map[p[0]], snt_token_ids_map[p[-1]]] for p in events]
    concs = [(conc[0], conc[-1]) for conc in concs_doc]
    concs = [[snt_ids_map[p[0]], snt_token_ids_map[p[0]], snt_token_ids_map[p[-1]]] for p in concs]
    return events, concs


def parse_pipeline_stage1(model, eval_features, data_type):
    model.eval()

    if data_type == 'modal':
        EDGE_LABEL_LIST = MODAL_EDGE_LABEL_LIST
    else:
        assert "temporal" in data_type
        EDGE_LABEL_LIST = TEMPORAL_EDGE_LABEL_LIST
    id2bio_map = id2EVENT_CONC_BIO

    list_of_predict_nodes = collections.defaultdict(list)
    for step, eval_feature in enumerate(eval_features):
        original_example = eval_feature["doc_example"]
        with torch.no_grad():
            stage1_tags = model(
                one_doc_example=eval_feature["doc_example"],
                lm_input=eval_feature["pretrained_lm_input"],
                is_training=False)
        if len(stage1_tags) != 0:
            stage1_tags = stage1_tags.detach().cpu().tolist()

        events, concs = get_doc_stage1_nodes(tags=stage1_tags, doc_example=original_example, id2bio=id2bio_map)
        list_of_predict_nodes[original_example].append([events, concs])

    stage1_string = write_stage1_parsed(list_of_predict_nodes, data_type)
    return stage1_string


def parse_end2end(model, eval_features, data_type, stage1_only=False, stage2_only=False):
    model.eval()

    if data_type == 'modal':
        EDGE_LABEL_LIST = MODAL_EDGE_LABEL_LIST
    else:
        assert "temporal" in data_type
        EDGE_LABEL_LIST = TEMPORAL_EDGE_LABEL_LIST
    id2bio_map = id2EVENT_CONC_BIO

    if stage1_only:
        list_of_predict_nodes = collections.defaultdict(list)
        for step, eval_feature in enumerate(eval_features):
            original_example = eval_feature["doc_example"]
            with torch.no_grad():
                stage1_scores, stage1_tags = model.decode_stage1(
                    one_doc_example=eval_feature["doc_example"],
                    lm_input=eval_feature["pretrained_lm_input"])
            if len(stage1_tags) != 0:
                stage1_tags = stage1_tags.detach().cpu().tolist()

            events, concs = get_doc_stage1_nodes(tags=stage1_tags, doc_example=original_example, id2bio=id2bio_map, scores=stage1_scores)
            list_of_predict_nodes[original_example].append([events, concs])

        stage1_string = write_stage1_parsed(list_of_predict_nodes, data_type)
        return stage1_string

    if stage2_only:
        parsed_child = []
        list_of_predict_edges = collections.defaultdict(list)

        for step, eval_feature in enumerate(eval_features):
            original_example = eval_feature["doc_example"]
            with torch.no_grad():
                label_scores = model.decode_stage2(
                    one_doc_example=eval_feature["doc_example"],
                    lm_input=eval_feature["pretrained_lm_input"],
                    gcn_input=eval_feature["gcn_input"],
                    use_pred_edges=False)
            if len(label_scores) == 0:
                continue

            _, predict_rel = label_scores.max(dim=-1)
            num_instances = predict_rel.tolist()
            for idx, p in enumerate(num_instances):
                p_parent_idx = int(predict_rel[idx] / len(EDGE_LABEL_LIST))
                p_label = predict_rel[idx] % len(EDGE_LABEL_LIST)
                p_parent = original_example.gold_edges[idx][p_parent_idx][0]
                c = original_example.gold_edges[idx][p_parent_idx][1]
                if (original_example.doc_id, c.ID) not in parsed_child:
                    parsed_child.append((original_example.doc_id, c.ID))
                else:
                    continue
                p_label = EDGE_LABEL_LIST[p_label]
                list_of_predict_edges[original_example].append((c.ID, c.label, p_parent.ID, p_label))

        stage2_string = write_stage2_parsed(list_of_predict_edges)
        return stage2_string

    if not stage1_only and not stage2_only:
        parsed_child = []
        list_of_predict_edges = collections.defaultdict(list)

        for step, eval_feature in enumerate(eval_features):
            with torch.no_grad():
                stage2_scores, stage1_tags, example_with_pred_edges = model(
                    one_doc_example=eval_feature["doc_example"],
                    lm_input=eval_feature["pretrained_lm_input"],
                    gcn_input=eval_feature["gcn_input"],
                    pred_nodes_ratio=1.,
                    is_training=False,
                    use_pred_edges=True)

            if len(stage2_scores) == 0:
                continue

            _, predict_rel = stage2_scores.max(dim=-1)
            num_instances = predict_rel.tolist()
            for idx, p in enumerate(num_instances):
                p_parent_idx = int(predict_rel[idx] / len(EDGE_LABEL_LIST))
                p_label = predict_rel[idx] % len(EDGE_LABEL_LIST)
                p_parent = example_with_pred_edges.pred_edges[idx][p_parent_idx][0]
                c = example_with_pred_edges.pred_edges[idx][p_parent_idx][1]
                if not (example_with_pred_edges.doc_id, c.ID) in parsed_child:
                    parsed_child.append((example_with_pred_edges.doc_id, c.ID))
                else:
                    continue
                p_label = EDGE_LABEL_LIST[p_label]
                list_of_predict_edges[example_with_pred_edges].append((c.ID, c.label, p_parent.ID, p_label))

        stage2_string = write_stage2_parsed(list_of_predict_edges)
        return stage2_string


def get_original_id(subtoken_ids, subtoken_map, sent_map, sent_token_map):
    parent_subtoken_s, parent_subtoken_e = subtoken_ids
    parent_token_s = subtoken_map[parent_subtoken_s]
    parent_token_e = subtoken_map[parent_subtoken_e]
    parent_sent = sent_map[parent_token_s]
    parent_token_in_sent_start = sent_token_map[parent_token_s]
    parent_token_in_sent_end = sent_token_map[parent_token_e]
    return parent_sent, parent_token_in_sent_start, parent_token_in_sent_end


def parse_span(model, eval_features):
    list_of_para_predict_parents = defaultdict(list)
    list_of_para_predict_labels = defaultdict(list)
    
    from data.reader_doc_qa import tag_paths_to_spans  # generate_parent_grand_p_from_bio_tag

    for step, batch in enumerate(eval_features):
        with torch.no_grad():
            bio_scores, rel_scores = model.decode(batch)
        for i, example in enumerate(batch["doc_example"]):
            original_example = example
            doc_id = original_example.doc_id

            subtoken_token_maps = batch["tokens"][i]
            context_start, context_end = batch["context_start_end"][i]
            context_sent_map = batch["flat_context_sent_map"][i]
            context_token_in_sent_map = batch["token_idx_in_sent_map"][i]

            bio_score, rel_score = bio_scores[i], rel_scores[i]
            bio = bio_score[context_start:context_end]
            rel = rel_score[context_start:context_end]

            assert len(bio) == len(rel) == len(subtoken_token_maps)

            bio_tags = [id2PARSING_BIO[t] for t in bio]
            rel_tags = [MODAL_EDGE_LABEL_LIST_SPAN_id2label[t] for t in rel]

            parents, grand_p, sec_parents, sec_grand_p = tag_paths_to_spans(bio_tags)
            parents = [(e[0], e[-1]) for e in parents]
            grand_p = [(e[0], e[-1]) for e in grand_p]

            sec_parents = [(e[0], e[-1]) for e in sec_parents]
            sec_grand_p = [(e[0], e[-1]) for e in sec_grand_p]


            parents_rel = [rel_tags[e[0]] for e in parents]
            grand_p_rel = [rel_tags[conc[0]] for conc in grand_p]

            sec_parent_rel = [rel_tags[e[0]] for e in sec_parents]
            sec_grand_p_rel = [rel_tags[conc[0]] for conc in sec_grand_p]

            parents = [(e[0], e[-1]) for e in parents]
            grand_parents = [(e[0], e[-1]) for e in grand_p]

            parents_original_ids = []
            sec_parent_original_ids = []
            for parent in parents:
                parent_sent, parent_token_in_sent_start, parent_token_in_sent_end = get_original_id(
                    parent, subtoken_map=subtoken_token_maps, sent_map=context_sent_map,
                    sent_token_map=context_token_in_sent_map)
                parents_original_ids.append([parent_sent, parent_token_in_sent_start, parent_token_in_sent_end])
            for sec_p in sec_parents:
                sec_parent_sent, sec_parent_token_in_sent_start, sec_parent_token_in_sent_end = get_original_id(
                    sec_p, subtoken_map=subtoken_token_maps, sent_map=context_sent_map,
                    sent_token_map=context_token_in_sent_map)
                sec_parent_original_ids.append([sec_parent_sent, sec_parent_token_in_sent_start, sec_parent_token_in_sent_end])

            grand_parents_original_ids = []
            sec_grand_parents_original_ids = []
            for g_parent in grand_parents:
                g_parent_sent, g_parent_token_in_sent_start, g_parent_token_in_sent_end = get_original_id(
                    g_parent, subtoken_map=subtoken_token_maps, sent_map=context_sent_map,
                    sent_token_map=context_token_in_sent_map)
                grand_parents_original_ids.append([
                    g_parent_sent, g_parent_token_in_sent_start, g_parent_token_in_sent_end])
            for s_g_p in sec_grand_p:
                s_g_parent_sent, s_g_parent_token_in_sent_start, s_g_parent_token_in_sent_end = get_original_id(
                    s_g_p, subtoken_map=subtoken_token_maps, sent_map=context_sent_map,
                    sent_token_map=context_token_in_sent_map)
                grand_parents_original_ids.append([
                    s_g_parent_sent, s_g_parent_token_in_sent_start, s_g_parent_token_in_sent_end])

            # Ignore sec_parent, sec_parent_rel..., are not used
            list_of_para_predict_parents[doc_id].append([
                original_example, parents_original_ids, grand_parents_original_ids,
                sec_parent_original_ids, sec_grand_parents_original_ids])
            list_of_para_predict_labels[doc_id].append([
                original_example, parents_rel, grand_p_rel, sec_parent_rel, sec_grand_p_rel])
    stage2_string = write_span_parsed(list_of_para_predict_parents, list_of_para_predict_labels)
    return stage2_string


def get_dev_stage2_f(dev_parsed, gold_file):
    gold_tuples = readin_tuples(gold_file, from_file=True)
    auto_tuples = readin_tuples(dev_parsed)
    macro_f, micro_f = eval_all(gold_tuples, auto_tuples, get_labeled_tups, labeled=True)
    return macro_f, micro_f


def parse_and_eval_pipeline_stage2(model, eval_features, data_type, gold_file):
    parsed_stage2 = parse_pipeline_stage2(model, eval_features, data_type=data_type)
    macro_f, micro_f = get_dev_stage2_f(parsed_stage2, gold_file)
    return macro_f, micro_f


def parse_and_eval_end2end(model, eval_features, data_type, gold_file):
    parsed_stage2 = parse_end2end(model, eval_features, data_type=data_type)
    e_macro_f, e_micro_f, conc_macro_f, conc_micro_f = get_dev_stage1_f(parsed_stage2, gold_file)
    macro_f, micro_f = get_dev_stage2_f(parsed_stage2, gold_file)
    return e_macro_f, e_micro_f, conc_macro_f, conc_micro_f, macro_f, micro_f


def parse_and_eval_span(model, eval_features, data_type, gold_file):
    parsed_stage2 = parse_span(model, eval_features)
    macro_f, micro_f = get_dev_stage2_f(parsed_stage2, gold_file)
    return macro_f, micro_f


def parse_and_eval_pipeline_stage1(model, eval_features, data_type, gold_file):
    parsed_stage2 = parse_pipeline_stage1(model, eval_features, data_type=data_type)
    e_macro_f, e_micro_f, conc_macro_f, conc_micro_f = get_dev_stage1_f(
        parsed_stage2, gold_file)
    return e_macro_f, e_micro_f, conc_macro_f, conc_micro_f
