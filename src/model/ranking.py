import torch
import torch.nn as nn
from torch.nn import Softmax
from transformers import AutoModel

from data.data_structures_modal import *
from model.shared import *
from model.edges_extraction import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 1

def encode_a_doc(bert, bert_dim, lm_input, pooling_method):
    """
    :param bert: bert model
    :param lm_input: input for bert
    :param pooling_method: take the max or average of subtokens in each token
    :return: num_token * bert_dim, encode a long document chunk by chunk, take the average of all the chunks
    to get the subtoken emb for each subtoken in the doc, then using the pooling method
    to get the token emb for each token in the doc.
    """
    doc_bert_output = []
    left_pads, right_pads = [], []
    num_token_per_doc = []
    for i, chunk_input_ids in enumerate(lm_input['input_ids']):
        # 1 * max_seq_len
        chunk_input_ids = torch.unsqueeze(chunk_input_ids, 0)
        chunk_atten_masks = torch.unsqueeze(lm_input['attention_mask'][i], 0)
        # 1 * max_seq_len * hidden
        chunk_outputs = bert(
            input_ids=chunk_input_ids,
            attention_mask=chunk_atten_masks,
            return_dict=True)
        # Sequence of hidden-states at the output of the last layer of the model
        # max_seq_len * hidden
        last_hidden_states = torch.squeeze(chunk_outputs["last_hidden_state"])

        # Ignore [PAD]
        active_token = lm_input['attention_mask'][i].view(-1) == 1
        last_hidden_states = last_hidden_states[active_token]

        # Ignore [CLS], [SEP]
        doc_bert_output.append(last_hidden_states[1:-1])

        # Pad to left
        # num_token_pad_to_left * hidden
        left_pads.append(torch.zeros(
            (lm_input["num_token_pad_to_left"][i], doc_bert_output[-1].shape[-1]),
            dtype=doc_bert_output[-1].dtype, device=doc_bert_output[-1].device))
        right_pads.append(torch.zeros(
            (lm_input["num_token_pad_to_right"][i], doc_bert_output[-1].shape[-1]),
            dtype=doc_bert_output[-1].dtype, device=doc_bert_output[-1].device))
        num_token_per_doc.append(doc_bert_output[-1].shape[0] +
                                 lm_input["num_token_pad_to_left"][i] + lm_input["num_token_pad_to_right"][i])
    # Assure each chunk has the same num of subtokens, i.e. each chunk is padded to the length of the doc
    for j in range(len(num_token_per_doc) - 1):
        assert num_token_per_doc[j] == num_token_per_doc[j + 1]

    # Pad each chunk, then take the average of the overlapping subtokens
    before_pooling_encoded_lst = []
    for j in range(len(doc_bert_output)):
        before_pooling_encoded_lst.append(torch.cat([left_pads[j], doc_bert_output[j], right_pads[j]], dim=0))

    # Combine all the chunks to get a completed doc
    before_pooling_encoded_lst = torch.stack(before_pooling_encoded_lst)

    if pooling_method == "max":
        pooled_encoded_lst = torch.max(before_pooling_encoded_lst, dim=0)[0]
    else:  # mean
        assert pooling_method == "average"
        pooled_encoded_lst = torch.sum(before_pooling_encoded_lst, dim=0)
        token_ratio = torch.unsqueeze(lm_input["token_ratio"], -1)
        token_ratio = token_ratio.expand(token_ratio.shape[0], pooled_encoded_lst.shape[-1])
        pooled_encoded_lst = pooled_encoded_lst * token_ratio

    # batch * doc_subtoken_num * bert_hidden, e.g. [1, 749, 768]
    doc_bert_output = torch.unsqueeze(pooled_encoded_lst, dim=0)

    # Get token representation from subtoken representation, which is the average of subtokens
    # idxs[-1]: length: token_num*token_len; masks[-1]: length: token_num*token_len
    idxs, masks, token_num, token_len = token_lens_to_idxs([lm_input["token_lens"]])

    # piece_idxs_tmp: batch * max_seq
    piece_idxs_tmp = lm_input['input_ids'][0].unsqueeze(0)
    batch_size = piece_idxs_tmp.shape[0]

    # [1, 3135, 768], 3135: 5 * num_token, 5 is max_num_of_subtoken_a_token_has
    # Convert idxs (a list) to a tensor that has the same dtype and device with input_ids
    idxs = piece_idxs_tmp.new(idxs).unsqueeze(-1).expand(batch_size, -1, bert_dim)  # + 1
    masks = doc_bert_output.new(masks).unsqueeze(-1)

    # [1, 3135, 768]
    doc_bert_output = torch.gather(doc_bert_output, 1, idxs) * masks
    doc_bert_output = doc_bert_output.view(batch_size, token_num, token_len, bert_dim)

    # batch * doc_token_num * bert_hidden, e.g. [1, 573, 768]
    # Sum over the token_len dim
    doc_bert_output = doc_bert_output.sum(2)
    return doc_bert_output


def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    # token_lens: suppose to be batch * num_tokens
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])

    idxs, masks = [], []
    seq_id_pad = 0
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [seq_id_pad] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([seq_id_pad] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


class PipeLineStage2(nn.Module):
    def __init__(self, config, params, edge_labels):
        super(PipeLineStage2, self).__init__()

        self.edge_label_set = edge_labels

        self.bert_dim = config.hidden_size
        self.bert = AutoModel.from_config(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_rgcn = params["use_rgcn"]
        self.parser = ModalDependencyGraphParser(config, params, edge_label_list=self.edge_label_set)

    def load_bert(self, model_name, cache_dir=None):
        print('Loading pre-trained BERT model {}'.format(model_name))
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def encode(self, lm_input, pooling="average"):
        return encode_a_doc(self.bert, self.bert_dim, lm_input, pooling)

    def forward(self, one_doc_example, lm_input, gcn_input, is_training=True, use_pred_edges=False):
        sequence_output = self.encode(lm_input)
        # batch * doc_token_num * bert_hidden
        sequence_output = self.bert_dropout(sequence_output)
        # doc_token_num * bert_hidden, as the batch size is 1
        sequence_output = torch.squeeze(sequence_output)
        assert sequence_output.shape[0] == sum([len(sent) for sent in one_doc_example.sentences_before_tokenization])

        if is_training:
            loss = self.parser(feats=sequence_output, one_doc_example=one_doc_example, gcn_inputs=gcn_input,
                               is_training=True, use_pred_edges=use_pred_edges)
            return loss
        else:
            scores = self.parser(feats=sequence_output, one_doc_example=one_doc_example, gcn_inputs=gcn_input,
                                 is_training=False, use_pred_edges=use_pred_edges)
            return scores


class PipeLineStage1(nn.Module):
    def __init__(self, config, params):
        super(PipeLineStage1, self).__init__()

        self.bert_dim = config.hidden_size
        self.bert = AutoModel.from_config(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = MentionExtractorEnd2end(config, tag_to_ix=EVENT_CONC_BIO2id, dropout_rate=params["stage1_dropout"])

    def load_bert(self, model_name, cache_dir=None):
        print('Loading pre-trained BERT model {}'.format(model_name))
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def encode(self, lm_input, pooling="average"):
        return encode_a_doc(self.bert, self.bert_dim, lm_input, pooling)

    def forward(self, one_doc_example, lm_input, is_training=True):
        # batch * doc_token_num * bert_hidden
        sequence_output = self.encode(lm_input)
        sequence_output = self.bert_dropout(sequence_output)
        # doc_token_num * bert_hidden, as the batch size is 1
        sequence_output = torch.squeeze(sequence_output)

        assert sequence_output.shape[0] == one_doc_example.bio_idx.shape[0]

        if is_training:
            loss_extraction, _, _ = self.extractor.compute_loss(sequence_output, one_doc_example)
            return loss_extraction
        else:
            stage1_scores, stage1_tags = self.extractor(sequence_output)
            return stage1_tags


class End2End(nn.Module):
    def __init__(self, config, params, edge_labels):
        super(End2End, self).__init__()

        self.edge_label_set = edge_labels

        self.bert_dim = config.hidden_size
        self.bert = AutoModel.from_config(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.extractor = MentionExtractorEnd2end(config, tag_to_ix=EVENT_CONC_BIO2id, dropout_rate=params["stage1_dropout"])
        self.parser = ModalDependencyGraphParser(config, params, edge_label_list=self.edge_label_set)
        self.use_rgcn = params["use_rgcn"]

    def load_bert(self, model_name, cache_dir=None):
        print('Loading pre-trained BERT model {}'.format(model_name))
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def encode(self, lm_input, pooling="average"):
        return encode_a_doc(self.bert, self.bert_dim, lm_input, pooling)

    def forward(self, one_doc_example, lm_input, gcn_input, pred_nodes_ratio=1.,
                is_training=True, use_pred_edges=True, regularize_adj=False):
        # batch * doc_token_num * bert_hidden
        sequence_output = self.encode(lm_input)
        sequence_output = self.bert_dropout(sequence_output)
        # doc_token_num * bert_hidden, as the batch size is 1
        sequence_output = torch.squeeze(sequence_output)

        assert sequence_output.shape[0] == one_doc_example.bio_idx.shape[0] \
               == sum([len(sent) for sent in one_doc_example.sentences_before_tokenization])

        if is_training:
            loss_extraction, stage1_tags, stage1_scores = self.extractor.compute_loss(sequence_output, one_doc_example)

            if use_pred_edges:
                example_with_pred_edges = generate_edges(
                    stage1_tags, id2EVENT_CONC_BIO, one_doc_example, self.edge_label_set,
                    pred_nodes_ratio, is_training=True)

                gcn_nodes, gcn_node_ids, rgcn_adjacency, sent_len_map = update_gcn_nodes(example_with_pred_edges)
                if regularize_adj:
                    softmax = nn.Softmax(dim=1)
                    normalized_scores = softmax(stage1_scores)
                    rgcn_adjacency = get_regularized_adj(normalized_scores, threshold,
                                                         example_with_pred_edges.pred_nodes,
                                                         gcn_node_ids,
                                                         rgcn_adjacency)
                else:
                    assert len(rgcn_adjacency.shape) == 3
                    rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor(
                        [sp.coo_matrix(rgcn_adjacency[i]) for i in range(rgcn_adjacency.shape[0])])
                gcn_input_with_pred_edges = {"gcn_nodes": gcn_nodes,
                             "gcn_node_id_map": gcn_node_ids,
                             "adj": rgcn_adjacency,
                             "sent_len_map": sent_len_map}
                for k, v in gcn_input_with_pred_edges.items():
                    if k == "adj":
                        v = [v]
                        gcn_input_with_pred_edges[k] = convert_3dsparse_to_4dsparse(v).to(device)
                    elif k == "gcn_node_id_map":
                        gcn_input_with_pred_edges[k] = v
                    else:
                        gcn_input_with_pred_edges[k] = torch.LongTensor(v).to(device)


                loss_parsing = self.parser(feats=sequence_output,
                                           one_doc_example=example_with_pred_edges,
                                           gcn_inputs=gcn_input_with_pred_edges,
                                           is_training=True,
                                           use_pred_edges=True)
            else:
                loss_parsing = self.parser(feats=sequence_output,
                                           one_doc_example=one_doc_example,
                                           gcn_inputs=gcn_input,
                                           is_training=True,
                                           use_pred_edges=False)
            loss = loss_extraction + loss_parsing
            return loss, loss_extraction, loss_parsing

        else:
            stage1_scores, stage1_tags = self.extractor(sequence_output)
            example_with_pred_edges = generate_edges(
                stage1_tags, id2EVENT_CONC_BIO, one_doc_example, self.edge_label_set,
                pred_nodes_ratio, is_training=False)

            gcn_nodes, gcn_node_ids, rgcn_adjacency, sent_len_map = update_gcn_nodes(example_with_pred_edges)
            if regularize_adj:
                softmax = nn.Softmax(dim=1)
                normalized_scores = softmax(stage1_scores)
                rgcn_adjacency = get_regularized_adj(normalized_scores, threshold,
                                                     example_with_pred_edges.pred_nodes,
                                                     gcn_node_ids,
                                                     rgcn_adjacency)
            else:
                assert len(rgcn_adjacency.shape) == 3
                rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor(
                    [sp.coo_matrix(rgcn_adjacency[i]) for i in range(rgcn_adjacency.shape[0])])

            gcn_input_with_pred_edges = {"gcn_nodes": gcn_nodes,
                         "gcn_node_id_map": gcn_node_ids,
                         "adj": rgcn_adjacency,
                         "sent_len_map": sent_len_map}
            for k, v in gcn_input_with_pred_edges.items():
                if k == "adj":
                    v = [v]
                    gcn_input_with_pred_edges[k] = convert_3dsparse_to_4dsparse(v).to(device)
                elif k == "gcn_node_id_map":
                    gcn_input_with_pred_edges[k] = v
                else:
                    gcn_input_with_pred_edges[k] = torch.LongTensor(v).to(device)

            stage2_scores = self.parser(sequence_output,
                                        example_with_pred_edges,
                                        # lm_input['input_ids'],
                                        gcn_inputs=gcn_input_with_pred_edges,
                                        is_training=False,
                                        use_pred_edges=True)
            return stage2_scores, stage1_tags, example_with_pred_edges

    def decode_stage1(self, one_doc_example, lm_input):
        sequence_output = self.encode(lm_input)
        sequence_output = self.bert_dropout(sequence_output)
        sequence_output = torch.squeeze(sequence_output)

        assert sequence_output.shape[0] == one_doc_example.bio_idx.shape[0]
        stage1_scores, stage1_tags = self.extractor(sequence_output)
        return stage1_scores, stage1_tags

    def decode_stage2(self, one_doc_example, lm_input, gcn_input, use_pred_edges):
        sequence_output = self.encode(lm_input)
        sequence_output = self.bert_dropout(sequence_output)
        sequence_output = torch.squeeze(sequence_output)
        assert sequence_output.shape[0] == one_doc_example.bio_idx.shape[0]
        stage2_scores = self.parser(feats=sequence_output,
                                    one_doc_example=one_doc_example,
                                    gcn_inputs=gcn_input,
                                    is_training=False,
                                    use_pred_edges=use_pred_edges)
        return stage2_scores
