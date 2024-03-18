import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as LOG_F
from torch.nn import MultiheadAttention

from data.data_structures_modal import *
from model.rgcn import RGCN_Layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some are inspired by https://github.com/nju-websoft/GLRE

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout_rate=0.):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MentionExtractorEnd2end(nn.Module):
    def __init__(self, config, tag_to_ix, dropout_rate=0.):
        super(MentionExtractorEnd2end, self).__init__()

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.bio_tag_scorer = MLP(config.hidden_size, self.tagset_size, dropout_rate)
        self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, feats, one_doc):
        # feats: doc_seq_len * bert_hidden
        scores = self.bio_tag_scorer(feats)
        _, tags = scores.max(dim=-1)
        # gold_tags: doc_seq_len
        gold_tags = one_doc.bio_idx.to(device)
        loss = self.loss_fct(scores, gold_tags)
        return loss, tags, scores

    def forward(self, feats):
        scores = self.bio_tag_scorer(feats)
        _, tags = scores.max(dim=-1)
        return scores, tags


def split_n_pad(nodes, section, pad=0, return_mask=False):
    """
    split tensor and pad
    :param nodes:
    :param section:
    :param pad:
    :return:
    """
    # nodes: num_tokens_in_batch * hidden;
    # sum(section.tolist()): total num tokens in doc/batch
    assert nodes.shape[0] == sum(section.tolist()), print(nodes.shape[0], sum(section.tolist()))
    # Split doc vec into sentence vec, num_sent * num_token_in_each_sent * hidden
    nodes = torch.split(nodes, section.tolist())

    # Pad to same length, num_sent * max_num_token * hidden
    nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)

    if not return_mask:
        return nodes
    else:
        max_v = max(section.tolist())
        temp_ = torch.arange(max_v).unsqueeze(0).repeat(nodes.size(0), 1).to(nodes)
        mask = (temp_ < section.unsqueeze(1))

        # mask = torch.zeros(nodes.size(0), max_v).to(nodes)
        # for index, sec in enumerate(section.tolist()):
        #    mask[index, :sec] = 1
        # assert (mask1==mask).all(), print(mask1)
        return nodes, mask


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.ignore = ignore

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze

        if pretrained:
            self.load_pretrained(pretrained, mapping)

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            trainable: (bool)

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        # if self.freeze:
        pret_embeds = torch.zeros((self.num_embeddings, self.embedding_dim))
        # else:
        # pret_embeds = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word in mapping.keys():
            if word in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word])
            elif word.lower() in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
        self.embedding = self.embedding.from_pretrained(pret_embeds, freeze=self.freeze) # , padding_idx=self.ignore

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds


class ModalDependencyGraphParser(nn.Module):
    def __init__(self, config, params, edge_label_list):
        super(ModalDependencyGraphParser, self).__init__()

        self.node_dim = config.hidden_size
        if params['type_dim'] > 0:
            self.has_type_emb = True
            self.node_dim += params['type_dim']
            self.type_embed = EmbedLayer(num_embeddings=3,
                                         embedding_dim=params['type_dim'],
                                         dropout=0.0)
        else:
            self.has_type_emb = False

        self.pair_dim = self.node_dim * 2
        self.ffn_hidden_dim = config.hidden_size

        if params['dist_dim'] > 0:
            self.has_dist_emb = True
            self.pair_dim += params['dist_dim'] * 3
            self.sent_embed_dir = EmbedLayer(num_embeddings=25,
                                              embedding_dim=params['dist_dim'],
                                              dropout=0.0,
                                              freeze=False,
                                              pretrained=None,
                                              mapping=None)
            self.dist_embed_dir = EmbedLayer(num_embeddings=50,
                                             embedding_dim=params['dist_dim'],
                                             dropout=0.0,
                                             freeze=False,
                                             pretrained=None,
                                             mapping=None)
        else:
            self.has_dist_emb = False

        if params["sent_vec"] in ["gcn", "orig"]:
            self.has_sent_vec = True
            self.sent_vec_type = params["sent_vec"]
        else:
            self.has_sent_vec = False

        self.use_rgcn = params["use_rgcn"]
        if self.use_rgcn:
            if params["rgcn_type"] == "rgcn":
                self.rgcn_layer = RGCN_Layer(
                    params, self.node_dim,
                    self.node_dim, 1, relation_cnt=params["rgcn_relation_cnt"])

        if params["self_atten_head_num"] > 0 and params["use_rgcn"]:
            self.has_self_atten = True
            self.self_atten = MultiheadAttention(
                embed_dim=self.node_dim, num_heads=params["self_atten_head_num"], batch_first=True)
            self.node_dim = self.node_dim + self.node_dim
            self.pair_dim += self.node_dim
        else:
            self.has_self_atten = False

        self.num_candidate_parent = params["num_candidate_parent"]
        self.labels = edge_label_list
        self.label_size = len(edge_label_list)

        self.mlp_label = MLP(self.pair_dim, self.ffn_hidden_dim)
        self.mlp_label_out = nn.Linear(self.ffn_hidden_dim, self.label_size)

        self.author_vec = nn.Parameter(torch.randn(self.node_dim))
        self.null_vec = nn.Parameter(torch.randn(self.node_dim))
        self.root_vec = nn.Parameter(torch.randn(self.node_dim))
        self.pad_vec = nn.Parameter(torch.randn(self.node_dim))
        if "before" in edge_label_list:
            # Temporal model
            self.dct_vec = nn.Parameter(torch.randn(self.node_dim))

        self.loss_fct = CrossEntropyLoss()

    def merge_tokens(self, token_info, feats, type="mean"):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them.
        token_info: [id, doc_start, doc_end, sent_id, node_type_id],
        1: doc_start column, 2: doc_end column
        @:param feats, seq_len * dim, e.g. 512*768
        """
        mentions = []
        for i in range(token_info.shape[0]):
            if type == "max":
                mention = torch.max(feats[token_info[i, 1]: token_info[i, 2], :], dim=-2)[0]
            else:  # mean
                mention = torch.mean(feats[token_info[i, 1]: token_info[i, 2], :], dim=-2)
            mentions.append(mention)
        mentions = torch.stack(mentions)
        return mentions

    def prepare_nodes_for_gcn(self, feats, sent_len_map, node_info):
        # Sent vec: average of token vec in sentence
        # Sent vec before average: num_sent_in_this_batch * max_sent_len * hidden
        # Sent vec after average: num_sent_in_this_batch * hidden
        sent_node_vec = split_n_pad(feats, sent_len_map)
        sent_node_vec = torch.mean(sent_node_vec, dim=1)

        # Mention vec: average of token vec in mention, need: num_mention * hidden
        # token_info: num_token * 5, each token: [id, doc_start, doc_end, sent_id, node_type_id]
        token_info = node_info[:(node_info.shape[0]-len(sent_len_map))]
        mention_node_vec = self.merge_tokens(token_info, feats)

        return mention_node_vec, sent_node_vec

    def forward(self, feats, one_doc_example,
                gcn_inputs, is_training=True, use_pred_edges=False):
        if not use_pred_edges:
            edges_using = one_doc_example.gold_edges
            gold_one_hot = one_doc_example.gold_one_hot
        else:
            edges_using = one_doc_example.pred_edges
            gold_one_hot = one_doc_example.pred_one_hot
        sequence_output = feats

        if len(edges_using) == 0:
            if is_training is True:
                return 0.
            else:
                return []

        # num_mention_nodes * hidden, num_sent_nodes * hidden
        mention_nodes, sent_nodes = self.prepare_nodes_for_gcn(
            sequence_output, gcn_inputs["sent_len_map"], gcn_inputs["gcn_nodes"])
        # (num_mention_nodes + num_sent_nodes) * hidden
        nodes_for_gcn = torch.cat([mention_nodes, sent_nodes], dim=0)
        assert nodes_for_gcn.shape[0] == gcn_inputs["gcn_nodes"].shape[0]

        if self.has_type_emb:
            nodes_for_gcn = torch.cat(
                (nodes_for_gcn, self.type_embed(gcn_inputs["gcn_nodes"][:, -1])), dim=1)

        if self.use_rgcn:
            # batch * num_nodes[mention_nodes + sentence_nodes] * hidden, e.g. [1, 102, 768]
            nodes_for_gcn = torch.unsqueeze(nodes_for_gcn, 0)
            # adj: batch * num_edge_types * num_nodes * num_nodes, e.g. [1, 3, 102, 102]
            nodes, _ = self.rgcn_layer(nodes_for_gcn, gcn_inputs['adj'])
            # num_nodes * hidden, as batch = 1, e.g. [102, 768]
            nodes = torch.squeeze(nodes, 0)
        else:
            # num_nodes * hidden
            nodes = nodes_for_gcn

        if self.has_self_atten:
            # version1: query: bert output; key and value: gcn output
            self_attented_nodes, attn_output_weights = self.self_atten(
                 query=nodes_for_gcn,
                 key=torch.unsqueeze(nodes, 0),
                 value=torch.unsqueeze(nodes, 0))
            nodes = torch.cat([nodes, torch.squeeze(self_attented_nodes, 0)], dim=-1)

        if self.has_dist_emb:
            child_sent_ids, parent_sent_ids, pair_distance = get_sent_distance_ids(edges_using, self.num_candidate_parent)
            child_sent_ids = torch.tensor(child_sent_ids, dtype=torch.int64).to(device)
            parent_sent_ids = torch.tensor(parent_sent_ids, dtype=torch.int64).to(device)
            pair_distance = torch.tensor(pair_distance, dtype=torch.int64).to(device)

            # [20, 16, 20], num_child * num_candidate_parent * sent_emb_hidden
            child_sent_embed = self.sent_embed_dir(child_sent_ids)
            parent_sent_embed = self.sent_embed_dir(parent_sent_ids)
            dist_embed = self.dist_embed_dir(pair_distance)

        if self.has_sent_vec:
            if self.sent_vec_type == "gcn":
                assert self.use_rgcn
                sent_vecs = nodes[-len(gcn_inputs["sent_len_map"]):]
            else:
                assert self.sent_vec_type == "orig"
                sent_vecs = sent_nodes #nodes_for_gcn[-len(gcn_inputs["sent_len_map"]):]

            sent_vecs_pad_meta = torch.cat([sent_vecs,
                                        torch.unsqueeze(self.pad_vec, 0),
                                        torch.unsqueeze(self.null_vec, 0),
                                        torch.unsqueeze(self.author_vec, 0),
                                        torch.unsqueeze(self.root_vec, 0)])
            sent_id_map_tmp = {-2:-4, -5:-3, -3:-2, -1:-1}
            sent_vecs = sent_vecs_pad_meta
            sent_vecs = torch.unsqueeze(sent_vecs, 0)

            child_sent_ids = [child[0][1].snt_index_in_doc for child in edges_using]
            child_sent_ids = [sent_id_map_tmp[c] if c in sent_id_map_tmp else c for c in child_sent_ids]
            child_sent_ids = torch.tensor(child_sent_ids , dtype=torch.int64)

            # num_child * num_candidate_parent
            child_sent_ids = torch.unsqueeze(child_sent_ids, dim=-1).expand(len(edges_using), self.num_candidate_parent)
            # [1, num_child, num_candidate_parent, hidden]: [1, 20, 16, 768]
            child_sent_vec = sent_vecs[:, child_sent_ids]

            # num_child * num_candidate_p: [20, 16]
            parent_sent_ids = [[c_p_pair[0].snt_index_in_doc for c_p_pair in child] for child in edges_using]
            parent_sent_ids = [[sent_id_map_tmp[p] if p in sent_id_map_tmp else p for p in child] for child in parent_sent_ids]
            parent_sent_ids = torch.tensor(parent_sent_ids, dtype=torch.int64)
            parent_sent_vecs = sent_vecs[:, parent_sent_ids]

        # Mention nodes only, e.g. [77, 768]
        nodes = nodes[:(gcn_inputs["gcn_nodes"].shape[0] - len(gcn_inputs["sent_len_map"]))]
        # assert len(gcn_inputs["gcn_node_id_map"]) == nodes.shape[0]

        # (num_mention_nodes + 4) * hidden, e.g. [81, 768]
        nodes_pad_meta = torch.cat([nodes,
                                    torch.unsqueeze(self.pad_vec, 0),
                                    torch.unsqueeze(self.null_vec, 0),
                                    torch.unsqueeze(self.author_vec, 0),
                                    torch.unsqueeze(self.root_vec, 0)])
        num_mention_nodes = nodes.shape[0]
        gcn_inputs["gcn_node_id_map"]["-2_-2_-2"] = num_mention_nodes
        gcn_inputs["gcn_node_id_map"]["-5_-5_-5"] = num_mention_nodes + 1
        gcn_inputs["gcn_node_id_map"]["-3_-3_-3"] = num_mention_nodes + 2
        gcn_inputs["gcn_node_id_map"]["-1_-1_-1"] = num_mention_nodes + 3
        assert gcn_inputs["gcn_node_id_map"]["-1_-1_-1"] == len(nodes_pad_meta) - 1
        assert len(gcn_inputs["gcn_node_id_map"]) == len(nodes_pad_meta)
        # 1 * (num_nodes + meta_nodes) * hidden
        nodes_pad_meta = torch.unsqueeze(nodes_pad_meta, 0)

        child_ids = torch.tensor(
            [gcn_inputs["gcn_node_id_map"][child[0][1].ID] for child in edges_using], dtype=torch.int64)
        num_child = child_ids.shape[0]
        # num_child * num_candidate_parent
        child_ids = torch.unsqueeze(child_ids, dim=-1).expand(num_child, self.num_candidate_parent)

        # [1, num_child, num_candidate_parent, hidden]: [1, 20, 16, 768]
        child_vecs = nodes_pad_meta[:, child_ids]

        # num_child * num_candidate_p: [20, 16]
        parent_ids = torch.tensor(
            [[gcn_inputs["gcn_node_id_map"][c_p_pair[0].ID] for c_p_pair in child] for child in edges_using],
            dtype=torch.int64)
        # [1, num_child, num_candidate_parent, hidden]: [1, 20, 16, 768]
        parent_vecs = nodes_pad_meta[:, parent_ids]

        if self.has_sent_vec:
            assert child_sent_vec.shape == child_vecs.shape
            assert parent_sent_vecs.shape == parent_vecs.shape

            child_vecs = torch.cat([child_vecs, child_sent_vec], dim=-1)
            child_vecs = self.child_sent_mlp(child_vecs)
            parent_vecs = torch.cat([parent_vecs, parent_sent_vecs], dim=-1)
            parent_vecs=self.parent_sent_mlp(parent_vecs)

        if self.has_dist_emb:
            # [1, 20, 16, 768 + 20]
            child_vecs = torch.cat([child_vecs, torch.unsqueeze(child_sent_embed, 0)], dim=-1)
            parent_vecs = torch.cat([parent_vecs, torch.unsqueeze(parent_sent_embed, 0)], dim=-1)

        # pair repr: [1, num_child, num_candidate_parent, hidden*2]: [1, 20, 16, (768 + 20) * 2]
        child_parent_vecs = torch.cat([child_vecs, parent_vecs], dim=-1)
        child_parent_vecs = torch.squeeze(child_parent_vecs, 0)  # [num_child, num_candidate_parent, hidden*2]

        if self.has_dist_emb:  # [1, 20, 16, (768 + 20) * 2 + 20 ]
            child_parent_vecs = torch.cat([child_parent_vecs, dist_embed], dim=-1)

        # [num_child, num_candidate_parent, hidden]: [20, 16, 768]
        child_parent_scores = self.mlp_label(child_parent_vecs)
        # [num_child, num_candidate_parent, 5]: [20, 16, 5]
        child_parent_scores = self.mlp_label_out(child_parent_scores)
        # child_parent_scores_for loss:  num_child * (num_candidate_parent*5): torch.Size([20, 80])
        child_parent_scores = child_parent_scores.view(-1, (self.num_candidate_parent*self.label_size))

        if is_training is False:
            if len(child_parent_scores) == 0:
                return []
            else:
                return child_parent_scores.detach().cpu()  # num_child * 80 (16*5)
        else:
            # gold_rel: torch.Size([20, 80])
            # gold_rel = torch.tensor(gold_one_hot).view(num_child, -1)
            gold_rel = torch.tensor(np.array(gold_one_hot)).view(num_child, -1)
            # num_child: torch.Size([20])
            _, gold_rel = gold_rel.max(dim=1)
            gold_rel = gold_rel.to(device)
            loss = self.loss_fct(child_parent_scores, gold_rel)
            return loss


def get_sent_distance_ids(edges, num_candidate_parent):
    # edges: num_child * num_candidate_parent * 3, 3 is a tuple size, i.e. (p, c, l)
    child_sent_ids = [int(child[0][1].snt_index_in_doc) for child in edges]
    child_sent_ids = [min(c, 20) for c in child_sent_ids]  # Ignore sent ids larger than 20
    map_tmp = {-1: 21, -2: 22, -3: 23, -5: 24}
    child_sent_ids = [map_tmp[c] if c < 0 else c for c in child_sent_ids]
    for c in child_sent_ids:
        assert c in list(range(25))

    parent_sent_ids = [[int(c_p_pair[0].snt_index_in_doc) for c_p_pair in child] for child in edges]
    parent_sent_ids = [[min(p, 20) for p in parents] for parents in parent_sent_ids]
    parent_sent_ids = [[map_tmp[p] if p < 0 else p for p in parents] for parents in parent_sent_ids]

    for parents in parent_sent_ids:
        for p in parents:
            assert p in list(range(25))

    child_sent_ids = [[c] * num_candidate_parent for c in child_sent_ids]

    pair_distance = []
    for child, parent in zip(child_sent_ids, parent_sent_ids):
        dis = []
        for c, p in zip(child, parent):
            assert c >= 0
            assert p >= 0
            d = c - p
            if d < 0:  # Child is before parent in the document
                d += 50
            dis.append(d)  # d in [0, 49]
            # else:
            #     dis.append(24)
        pair_distance.append(dis)

    return child_sent_ids, parent_sent_ids, pair_distance
