import sys
import codecs
import argparse
import re


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gold_file", help="gold file")
    arg_parser.add_argument("--parsed_file", help="parsed file")
    arg_parser.add_argument("--unlabeled", action="store_true", default=False)
    arg_parser.add_argument("--rel_only", action="store_true", default=False)
    arg_parser.add_argument("--eval_stage1", action="store_true", default=False)
    return arg_parser


def readin_tuples(filename):
    lines = codecs.open(filename, 'r', 'utf-8').readlines()
    edge_tuples = []
    mode = None
    # if 'doc id' in lines[0]:
    doc_ids = []
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                if 'doc id' in line:
                    doc_ids.append(re.search(r'\d+', line, flags=0).group())
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            edge = line.strip().split('\t')
            assert len(edge) == 4  # in [2, 4]
            child, child_label, parent, link_label = edge
            edge_tuples[-1].append((child, parent, link_label))
    return edge_tuples, doc_ids


def readin_stage1_tuples(filename):
    lines = codecs.open(filename, 'r', 'utf-8').readlines()
    edge_tuples = []
    mode = None
    doc_ids = []
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                if 'doc id' in line:
                    doc_ids.append(re.search(r'\d+', line, flags=0).group())
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            edge = line.strip().split('\t')
            assert len(edge) in [2, 4]
            if len(edge) == 4:
                child, child_label, parent, link_label = edge
            else:
                child, child_label = edge
            if int(child.split('_')[0]) < 0:
                #   print('meta nodes: ', child)
                continue
            edge_tuples[-1].append((child, child_label))
    return edge_tuples, doc_ids


def get_labeled_tups(tups):
    # tup: child, parent, rel
    tups = set([(tup[0], tup[1], tup[2]) for tup in tups])
    return tups


def get_unlabeled_tups(tups):
    # tup: child, parent
    tups = set([(tup[0], tup[1]) for tup in tups])
    return tups


def get_rel_only_tups(tups):
    # tup: child, rel
    return set([(tup[0], tup[2]) for tup in tups])


def get_stage1_labeled_tuple_set(tups):
    # tup: child, child_label
    tup_set = set([(tup[0], tup[1]) for tup in tups])
    return tup_set


def get_stage1_unlabeled_tuple_set(tups):
    # tup: child
    tup_set = set([tup[0] for tup in tups])
    return tup_set


def get_event_tups(tups):
    assert len(tups[-1]) == 2
    assert tups[-1][-1][0] in ['E', 'C']
    return set([tup[0] for tup in tups if tup[1].startswith('E')])


def get_te_tups(tups):
    assert len(tups[-1]) == 2
    assert tups[-1][-1][0] in ['E', 'C']
    return set([tup[0] for tup in tups if not tup[1].startswith('E')])


def compute_p_r_f(gold_tuples, auto_tuples, get_tups):
    counts = []
    scores = []
    for i, (gtups, atups) in enumerate(zip(gold_tuples, auto_tuples)):
        gtups = get_tups(gtups)
        atups = get_tups(atups)
        true_positive = len(gtups.intersection(atups))
        false_positive = len(atups.difference(gtups))
        false_negative = len(gtups.difference(atups))

        if false_positive + true_positive == 0 or false_negative + true_positive == 0:
            p, r, f = 0, 0, 0
        else:
            p = true_positive / (true_positive + false_positive)
            r = true_positive / (true_positive + false_negative)
            f = 2 * p * r / (p + r) if p + r != 0 else 0

        print('test doc {}: true_p = {}, false_p = {}, false_n = {}, f = {}'.format(
            i, true_positive, false_positive, false_negative, f))

        counts.append((true_positive, false_positive, false_negative))
        scores.append((p, r, f))

    # macro average
    if len(scores) == 0:
        macro_p = macro_r = macro_f = 0
    else:
        macro_p = sum([score[0] for score in scores]) / len(scores)
        macro_r = sum([score[1] for score in scores]) / len(scores)
        macro_f = sum([score[2] for score in scores]) / len(scores)

    # micro average
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])

    if true_p + false_p == 0:
        print('warning! zero true_p, false_p')
        micro_p = 0
    else:
        micro_p = true_p / (true_p + false_p)

    if true_p + false_n == 0:
        print('warning! zero true_p, false_n')
        micro_r = 0
    else:
        micro_r = true_p / (true_p + false_n)

    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r != 0 else 0

    return macro_p, macro_r, macro_f, micro_p, micro_r, micro_f


def eval_all(gold_tuples, auto_tuples, get_tups, labeled):
    macro_p, macro_r, macro_f, micro_p, micro_r, micro_f = compute_p_r_f(gold_tuples, auto_tuples, get_tups)
    if not labeled:
        print('unlabeled macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(macro_p, macro_r, macro_f))
        print('unlabeled micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(micro_p, micro_r, micro_f))
    else:
        print('labeled macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(macro_p, macro_r, macro_f))
        print('labeled micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(micro_p, micro_r, micro_f))
    return macro_f, micro_f


def eval_stage1(gold_tuples, auto_tuples, get_event_tups, get_te_tups):
    e_macro_p, e_macro_r, e_macro_f, e_micro_p, e_micro_r, e_micro_f = \
        compute_p_r_f(gold_tuples, auto_tuples, get_event_tups)

    te_macro_p, te_macro_r, te_macro_f, te_micro_p, te_micro_r, te_micro_f = \
        compute_p_r_f(gold_tuples, auto_tuples, get_te_tups)

    print('e macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(e_macro_p, e_macro_r, e_macro_f))
    print('e micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(e_micro_p, e_micro_r, e_micro_f))

    print('te macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(te_macro_p, te_macro_r, te_macro_f))
    print('te micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(te_micro_p, te_micro_r, te_micro_f))
    return e_macro_f, e_micro_f, te_macro_f, te_micro_f


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if not args.eval_stage1:
        gold_tuples, gold_doc_ids = readin_tuples(args.gold_file)
        auto_tuples, auto_doc_ids = readin_tuples(args.parsed_file)
        assert gold_doc_ids == auto_doc_ids
        #assert len(auto_doc_ids) == len(set(auto_doc_ids))
        # if "chinese" in args.gold_file:
        #     assert len(auto_doc_ids) == 30
        # else:
        #     assert len(auto_doc_ids) == 32
        # assert len(gold_tuples) == len(auto_tuples)
        # assert len(gold_doc_ids), len(auto_doc_ids))
        if args.unlabeled:
            print('warning! unlabeled eval')
            eval_all(gold_tuples, auto_tuples, get_unlabeled_tups, labeled=False)
        elif args.rel_only:
            print('warning! rel only eval')
            eval_all(gold_tuples, auto_tuples, get_rel_only_tups, labeled=False)
        else:
            eval_all(gold_tuples, auto_tuples, get_labeled_tups, labeled=True)
    else:
        gold_tuples, gold_doc_ids = readin_stage1_tuples(args.gold_file)
        auto_tuples, auto_doc_ids = readin_stage1_tuples(args.parsed_file)
        assert gold_doc_ids == auto_doc_ids
        # if "chinese" in args.gold_file:
        #     assert len(auto_doc_ids) == 30
        # else:
        #     assert len(auto_doc_ids) == 32
        #assert len(auto_doc_ids) == len(set(auto_doc_ids))
        eval_stage1(gold_tuples, auto_tuples, get_event_tups, get_te_tups)
