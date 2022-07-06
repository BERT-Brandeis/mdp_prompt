UNK_word = '<UNK>'
UNK_label = '<UNK>'
PAD_word = 'PADDING'
PAD_label = '<PAD>'
ROOT_word = 'ROOT'
ROOT_label = '<ROOT>'
AUTHOR_word = 'AUTHOR'
AUTHOR_label = '<AUTHOR>'
NULL_CONCEIVER_word = 'NULL'
NULL_CONCEIVER_label = '<NULL>'
DCT_word = 'DCT'
DCT_label = '<DCT>'


#MODAL_EDGE_LABEL_LIST = [
#'pos',
#'neg',
#'pp',
#'pn',
#'Depend-on']
MODAL_EDGE_LABEL_LIST = [
'pos',
'neg',
'neg_prt',
'neg_neut',
'pp',
'pn',
'Depend-on'
]

TEMPORAL_EDGE_LABEL_LIST = [
'before',
'after',
'overlap',
'included',
'Depend-on']

EVENT_CONC_BIO2id = {'b_e': 0, 'i_e': 1, 'b_c': 2, 'i_c': 3, 'o': 4, 'b_t': 5, 'i_t': 6}
id2EVENT_CONC_BIO = {v: k for k, v in EVENT_CONC_BIO2id.items()}
EVENT_CONC_BIO = ['b_e', 'i_e', 'b_c', 'i_c', 'o', 'b_t', 'i_t']

PARSING_BIO2id = {'b_c': 0, 'i_c': 1, 'b_g': 2, 'i_g': 3, 'o': 4}
id2PARSING_BIO = {v: k for k, v in PARSING_BIO2id.items()}
PARSING_BIO = ['b_c', 'i_c', 'b_g', 'i_g', 'o']

MODAL_EDGE_LABEL_LIST_SPAN = MODAL_EDGE_LABEL_LIST + ['NA']
MODAL_EDGE_LABEL_LIST_SPAN_label2id = {label: i for i, label in enumerate(MODAL_EDGE_LABEL_LIST_SPAN)}
MODAL_EDGE_LABEL_LIST_SPAN_id2label = {i: label for i, label in enumerate(MODAL_EDGE_LABEL_LIST_SPAN)}


class Node:
    def __init__(self, snt_index_in_doc=-1, start_word_index_in_snt=-1, end_word_index_in_snt=-1, node_index_in_doc=-1,
                 start_word_index_in_doc=-1, end_word_index_in_doc=-1, words=ROOT_word, label=ROOT_label):
        self.snt_index_in_doc = snt_index_in_doc
        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt
        self.node_index_in_doc = node_index_in_doc
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc

        self.words = words
        self.label = label

        self.is_DCT = self.words == DCT_word
        self.ID = '_'.join([str(snt_index_in_doc),
                            str(start_word_index_in_snt), str(end_word_index_in_snt)])

    def __str__(self):
        return '\t'.join([self.ID, self.words, self.label])

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.ID == other.ID
        return False

    def __hash__(self):
        return hash(self.ID)


def get_root_node():
    return Node()


def get_author_node():
    return Node(-3, -3, -3, -3, -3, -3, AUTHOR_word, AUTHOR_label)


def get_null_conceiver_node():
    return Node(-5, -5, -5, -5, -5, -5, NULL_CONCEIVER_word, NULL_CONCEIVER_label)


def get_padding_node():
    return Node(-2, -2, -2, -2, -2, -2, PAD_word, PAD_label)


def is_author_node(node):
    return node.snt_index_in_doc == -3


def is_root_node(node):
    return node.snt_index_in_doc == -1


def is_padding_node(node):
    return node.snt_index_in_doc == -2


def is_null_conceiver_node(node):
    return node.snt_index_in_doc == -5
