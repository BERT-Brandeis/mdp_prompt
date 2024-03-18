import codecs
from data.data_structures_modal import *


eng_max_padded_candidate_length = 16
chn_max_padded_candidate_length = 40
eng_max_sent_distance = 5
chn_max_sent_distance = (1000, 3)
eng_e2e_sent_distance = 0
chn_e2e_sent_distance = 4


def get_word_index_in_doc(snt_list, snt_index_in_doc, word_index_in_snt):
    index = 0
    if snt_index_in_doc < 0:
        return snt_index_in_doc
    for i, snt in enumerate(snt_list):
        if i < snt_index_in_doc:
            index += len(snt)
        else:
            break
    return index + word_index_in_snt


def create_snt_edge_lists(doc):
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            assert mode in ["SNT_LIST", "EDGE_LIST"]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            one_edge = line.strip().split()
            if len(one_edge) == 2:
                # Stage1 output
                one_edge.extend([None, None])
            assert len(one_edge) == 4
            edge_list.append(one_edge)

    def sort_edge_by_node_id_in_doc(edge):
        node, _, _, _ = edge
        node_snt, node_token_s, node_token_e = [int(i) for i in node.split('_')]
        node_idx_in_doc = get_word_index_in_doc(snt_list, node_snt, node_token_s)
        return node_idx_in_doc

    if len(edge_list) > 0:
        if not edge_list[0][0] == '-3_-3_-3':
            edge_list = [['-3_-3_-3', 'Conceiver', None, None]] + edge_list

    edge_list.sort(key=sort_edge_by_node_id_in_doc)

    # for edge in edge_list:
    #     print(edge)
    # print()

    return snt_list, edge_list


def create_node_list(snt_list, edge_list):
    node_list = []
    for i, edge in enumerate(edge_list):
        child, c_label, _, _ = edge  # Parent isn't used here
        c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]
        if c_snt >= 0:
            # print(' '.join(snt_list[c_snt]))
            c_words = ' '.join(snt_list[c_snt][c_start:c_end + 1])
        else:
            assert c_snt == -3
            c_words = AUTHOR_word
            # if c_snt == -3:
            #     c_words = AUTHOR_word

        if '-' in c_label:
            c_label, _ = c_label.split("-")
        assert c_label in ["Event", "Conceiver"]

        c_node = Node(snt_index_in_doc=c_snt,
                      start_word_index_in_snt=c_start,
                      end_word_index_in_snt=c_end,
                      node_index_in_doc=i,
                      start_word_index_in_doc=get_word_index_in_doc(snt_list, c_snt, c_start),
                      end_word_index_in_doc=get_word_index_in_doc(snt_list, c_snt, c_end),
                      words=c_words,
                      label=c_label)
        node_list.append(c_node)

    return node_list


def check_example_contains_gold_parent(example):
    for tup in example:
        if tup[2] != 'NO_EDGE' and tup[2] != None:
            assert tup[2] in ['pos', 'pp', 'pn', 'neg', 'neg_prt',
                              'neg_neut', 'Depend-on']
            return True
    return False


def make_one_doc_training_data(doc, max_sent_distance):
    """
    training_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'pos'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # Create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # Create node_list
    node_list = create_node_list(snt_list, edge_list)

    # Create training example list
    training_example_list = []

    root_node = get_root_node()
    padding_node = get_padding_node()
    null_conceiver_node = get_null_conceiver_node()

    for i, edge in enumerate(edge_list):
        child_tmp, _, parent, label = edge
        child_node = node_list[i]
        assert child_tmp == child_node.ID

        example = choose_candidates_padded(
            node_list, root_node, padding_node, null_conceiver_node,
            child_node, parent, label, max_sent_distance)
        if check_example_contains_gold_parent(example):
            training_example_list.append(example)
        else:
            # doc_name = doc[0].split(':')[1]
            print('WARNING: Gold parent not included for edge {} in document {}'.format(edge, doc[0]))

    # for child in training_example_list:
    #     print('check_one_child...')
    #     for item in child:
    #        print(item[0], item[1], item[2])
    #     print()

    return [snt_list, training_example_list, node_list]


def choose_candidates_padded(node_list, root_node, padding_node, null_conceiver_node,
                             child, parent_ID, label, max_sent_distance):
    candidates = []
    if not isinstance(max_sent_distance, int):
        # for Chinese data
        max_sent_before, max_sent_after = max_sent_distance
        # assert max_sent_before == 1000
        max_padded_candidate_length = chn_max_padded_candidate_length
        e2e_sent_distance = chn_e2e_sent_distance
    else:
        # for English data
        max_sent_before = max_sent_distance
        max_sent_after = max_sent_distance
        max_padded_candidate_length = eng_max_padded_candidate_length
        e2e_sent_distance = eng_e2e_sent_distance
        # assert max_sent_after == max_sent_before == 5
        # assert max_padded_candidate_length == 16
        # assert e2e_sent_distance == 0

    # Always consider root
    candidates.append(get_candidate(child, root_node, parent_ID, label))

    # Get non-event candidates
    for candidate_node in node_list:
        if candidate_node.label.startswith('E'):
            continue
        if candidate_node.ID == child.ID:
            continue
        if candidate_node.snt_index_in_doc >= 0 and (
                child.snt_index_in_doc - candidate_node.snt_index_in_doc) > max_sent_before:
            continue
        if candidate_node.snt_index_in_doc >= 0 and (
                candidate_node.snt_index_in_doc - child.snt_index_in_doc) > max_sent_after:
            continue
        candidates.append(get_candidate(child, candidate_node, parent_ID, label))

    # Get event candidates
    for candidate_node in node_list:
        if not candidate_node.label.startswith('E'):
            continue
        # For e2e cases, assume child snt idx is always after parent
        if child.snt_index_in_doc - candidate_node.snt_index_in_doc > e2e_sent_distance:
            continue
        # For e2e cases, assume child idx is always after parent
        if candidate_node.node_index_in_doc > child.node_index_in_doc:
            continue
        if candidate_node.ID == child.ID:
            continue
        candidates.append(get_candidate(child, candidate_node, parent_ID, label))

    # Always consider null_conceiver
    candidates.append(get_candidate(child, null_conceiver_node, parent_ID, label))

    if len(candidates) < max_padded_candidate_length:
        padding_length = max_padded_candidate_length - len(candidates)
        for i in range(padding_length):
            candidates.append(get_candidate(child, padding_node, parent_ID, label))
    if len(candidates) > max_padded_candidate_length:
        candidates = candidates[:max_padded_candidate_length]
    assert len(candidates) == max_padded_candidate_length

    return candidates


def get_candidate(child, candidate_node, parent_ID, label):
    if label:
        # Training on labels, so either add label or 'NO_EDGE'
        if candidate_node.ID == parent_ID:
            return candidate_node, child, label
        else:
            return candidate_node, child, 'NO_EDGE'
    else:
        # Predicting labels with model, so add None here
        return candidate_node, child, None


def make_training_data(train_file, language=None):
    """ Given a file of multiple documents in ConLL-similar format,
    produce a list of training docs, each training doc is
    (1) a list of sentences in that document; and
    (2) a list of (parent_candidate, child_node, edge_label/no_edge) tuples
    in that document;
    (3) a list of nodes in that document
    """
    # from collections import defaultdict
    # max_edges = defaultdict(int)

    data = codecs.open(train_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    training_data = []
    doc_id_list = []
    for i, doc in enumerate(doc_list):
        doc_id = doc.split(':SNT_LIST')[0].strip('filename')
        doc_id_list.append(doc_id)
        # print("file name: ", doc_id)
        if language == 'chn':
            training_data.append(
                make_one_doc_training_data(doc, max_sent_distance=chn_max_sent_distance))
        else:
            assert language == 'eng'
            training_data.append(
                make_one_doc_training_data(doc, max_sent_distance=eng_max_sent_distance))

    assert len(doc_id_list) == len(set(doc_id_list)) == len(training_data)  # == len(doc_list)

    return training_data, doc_id_list



def make_one_doc_test_data(doc, max_sent_distance):
    doc = doc.strip().split('\n')

    # Create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # Create node_list
    node_list = create_node_list(snt_list, edge_list)

    # Create test instance list
    test_instance_list = []

    root_node = get_root_node()
    padding_node = get_padding_node()
    null_conceiver_node = get_null_conceiver_node()

    for c_node in node_list:
        instance = choose_candidates_padded(
            node_list, root_node, padding_node, null_conceiver_node, c_node,
            parent_ID=None, label=None, max_sent_distance=max_sent_distance)
        test_instance_list.append(instance)

    return [snt_list, test_instance_list, node_list]


def make_test_data(test_file, language=None):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []
    doc_id_list = []
    for doc in doc_list:
        doc_id = doc.split(':SNT_LIST')[0].strip('filename')

        while ':filename' in doc_id:
            # print(doc_id, doc_id.strip(':filename'))
            doc_id = doc_id.strip(':filename')
        doc_id_list.append(doc_id)

        if language == 'chn':
            test_data.append(
                make_one_doc_test_data(doc, max_sent_distance=chn_max_sent_distance))
        else:
            assert language == 'eng'
            test_data.append(
                make_one_doc_test_data(doc, max_sent_distance=eng_max_sent_distance))
    assert len(doc_id_list) == len(set(doc_id_list)) == len(test_data) == len(doc_list)
    # if language == 'chn':
    #     assert len(doc_id_list) == 30
    # else:
    #     assert len(doc_id_list) == 32

    return test_data, doc_id_list


def make_test_data_from_doc_lst(input_lst):
    """
    Assume the input is a list of doc,
    each doc is a list of sentences, each sentences is a list of tokens.
    """
    test_data = []
    doc_id_list = []
    for d_id, doc in enumerate(input_lst):
        doc_id_list.append('filename:<doc id=modal_doc_list_' + str(d_id) + '>')
        test_data.append([doc, [], []])
    assert len(doc_id_list) == len(set(doc_id_list)) == len(test_data)
    return test_data, doc_id_list




def generate_e_conc_from_bio_tag(bio_tag_list, data_type):
    events = []
    tes = []

    if data_type == 'modal':
        non_event_start = 'b_c'
        non_event_inter = 'i_c'
    else:
        assert "temporal" in data_type
        non_event_start = 'b_t'
        non_event_inter = 'i_t'

    for t_idx, t in enumerate(bio_tag_list):
        if t == 'b_e':
            one_e = [t_idx]
            if t_idx == len(bio_tag_list) - 1:
                # It's the last token
                one_e.append(t_idx)
                events.append(one_e)
                continue
            if bio_tag_list[t_idx + 1] != 'i_e':
                one_e.append(t_idx)
                events.append(one_e)
                continue
            for t_idx1, t1 in enumerate(bio_tag_list[t_idx + 1:]):
                if t1 == 'i_e':
                    one_e.append(t_idx1 + t_idx + 1)
                    if t_idx1 + t_idx + 1 == len(bio_tag_list) - 1:
                        events.append(one_e)
                else:
                    events.append(one_e)
                    break
        elif t == non_event_start:
            one_te = [t_idx]
            if t_idx == len(bio_tag_list) - 1:
                one_te.append(t_idx)
                tes.append(one_te)
                continue
            if bio_tag_list[t_idx + 1] != non_event_inter:
                one_te.append(t_idx)
                tes.append(one_te)
                continue
            for t_idx1, t1 in enumerate(bio_tag_list[t_idx + 1:]):
                if t1 == non_event_inter:
                    one_te.append(t_idx1 + t_idx + 1)
                    if t_idx1 + t_idx + 1 == len(bio_tag_list) - 1:
                        tes.append(one_te)
                else:
                    tes.append(one_te)
                    break
    return events, tes
