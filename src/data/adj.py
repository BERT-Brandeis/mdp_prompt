import numpy as np
import torch
import scipy.sparse as sp


node_type_map = {"Event": 0, "Conceiver": 1, "Sent": 2, "Meta_node": 3}


def get_gcn_nodes(mention_node_lst, doc_sent_lst):
    """
    Node: [id, doc_start, doc_end, sent_id, node_type_id]
    :return: [Mention Nodes; Sent Nodes]
    """
    gcn_nd = []
    nd_id_maps = {}
    non_abstract_node_lst = [nd for nd in mention_node_lst if nd.snt_index_in_doc >= 0]
    for id, nd in enumerate(non_abstract_node_lst):
        assert nd.end_word_index_in_doc >= nd.start_word_index_in_doc >= 0
        node_sent, node_doc_start, node_doc_end = nd.snt_index_in_doc, nd.start_word_index_in_doc, nd.end_word_index_in_doc
        if nd.label.startswith("E"):
            nd_type = node_type_map["Event"]
        else:
            assert nd.label.startswith("C")
            nd_type = node_type_map["Conceiver"]
        gcn_nd.append([id, node_doc_start, node_doc_end + 1, node_sent, nd_type])
        nd_id_maps[nd.ID] = id

    assert len(nd_id_maps) == len(non_abstract_node_lst) == len(gcn_nd)

    for s, sentence in enumerate(doc_sent_lst):
        gcn_nd += [[s, s, s, s, node_type_map["Sent"]]]
    return np.array(gcn_nd), nd_id_maps


def get_adj(nodes, mention_nodes_lst=None, return_sparse_tensor=True):
    """
    :param nodes: output of get_gcn_nodes(), i.e. a list of gcn nodes;
    :param mention_nodes_lst: original nodes, i.e. output of data_preparation_modal
    """
    n_id, mention_s, mention_e, sent_id, nd_type = 0, 1, 2, 3, 4

    xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
    # r_idx, c_idx = nodes[xv, n_id], nodes[yv, n_id]  # node id

    r_id, c_id = nodes[xv, nd_type], nodes[yv, nd_type]  # node type
    r_Sid, c_Sid = nodes[xv, sent_id], nodes[yv, sent_id]  # sent id
    # r_Ms, c_Ms = nodes[xv, mention_s], nodes[yv, mention_s]  # mention start
    # r_Me, c_Me = nodes[xv, mention_e], nodes[yv, mention_e]  # mention end

    # mm_c, mm_e, ss, ms = [0, 1, 2, 3]#[0, 1, 2]#, 3]
    mm, ss, ms = [0, 1, 2]
    num_edge_type = 3
    # num_edge_type * num_nodes * num_nodes
    rgcn_adjacency = np.full((num_edge_type, r_id.shape[0], r_id.shape[0]), 0.0)

    # mention-mention: link mentions that are in the same sentence
    # r_Sid == c_Sid: in same sentence
    rgcn_adjacency[mm] = np.where(
        np.logical_or(r_id == 0, r_id == 1, r_id == 3) & np.logical_or(c_id == 0, c_id == 1, c_id == 3)
        & (r_Sid == c_Sid), 1, rgcn_adjacency[mm])

    # rgcn_adjacency[mm_c] = np.where(
    #     np.logical_or(r_id == 0, r_id == 1) & (c_id == 1)
    #     & (r_Sid == c_Sid), 1, rgcn_adjacency[mm_c])
    # rgcn_adjacency[mm_c] = np.where(
    #     (r_id == 1) & np.logical_or(c_id == 0, c_id == 1)
    #     & (r_Sid == c_Sid), 1, rgcn_adjacency[mm_c])
    # # self loop
    # rgcn_adjacency[mm_c] = np.where(
    #     np.logical_or(r_id == 0, r_id == 1, r_id == 3) & np.logical_or(c_id == 0, c_id == 1, c_id == 3) &
    #     (r_Ms == c_Ms) & (r_Me == c_Me) & (r_Sid == c_Sid), 1, rgcn_adjacency[mm_c])
    #
    # rgcn_adjacency[mm_e] = np.where((r_id == 0) & (c_id == 0) & (r_Sid == c_Sid), 1, rgcn_adjacency[mm_e])

    # sentence-sentence (direct + indirect)
    rgcn_adjacency[ss] = np.where((r_id == 2) & (c_id == 2), 1, rgcn_adjacency[ss])

    # mention-sentence: link mentions to the sentence they are in
    rgcn_adjacency[ms] = np.where(np.logical_or(r_id == 0, r_id == 1, r_id == 3) & (c_id == 2) & (r_Sid == c_Sid), 1,
                                  rgcn_adjacency[ms])  # belongs to sentence
    rgcn_adjacency[ms] = np.where((r_id == 2) & np.logical_or(c_id == 0, c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1,
                                  rgcn_adjacency[ms])  # inverse

    if return_sparse_tensor:
        rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor(
            [sp.coo_matrix(rgcn_adjacency[i]) for i in range(num_edge_type)])

    return rgcn_adjacency


def sparse_mxs_to_torch_sparse_tensor(sparse_mxs):
    """
    Convert a list of scipy sparse matrix to a torch sparse tensor.
    :param sparse_mxs: [sparse_mx] adj
    :return:
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[0])
    b_index = []
    row_index = []
    col_index = []
    value = []
    for index, sparse_mx in enumerate(sparse_mxs):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        b_index.extend([index] * len(sparse_mx.row))
        row_index.extend(sparse_mx.row)
        col_index.extend(sparse_mx.col)
        value.extend(sparse_mx.data)
    indices = torch.from_numpy(
        np.vstack((b_index, row_index, col_index)).astype(np.int64))
    values = torch.FloatTensor(value)
    shape = torch.Size([len(sparse_mxs), max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)


def convert_3dsparse_to_4dsparse(sparse_mxs):
    """
    :param sparse_mxs: [3d_sparse_tensor]
    :return:
    """
    max_shape = 0
    for mx in sparse_mxs:
        max_shape = max(max_shape, mx.shape[1])
    b_index = []
    indexs = []
    values = []
    for index, sparse_mx in enumerate(sparse_mxs):
        indices_ = sparse_mx._indices()
        values_ = sparse_mx._values()
        b_index.extend([index] * values_.shape[0])
        indexs.append(indices_)
        values.append(values_)
    indexs = torch.cat(indexs, dim=-1)
    b_index = torch.as_tensor(b_index)
    b_index = b_index.unsqueeze(0)
    indices = torch.cat([b_index, indexs], dim=0)
    values = torch.cat(values, dim=-1)
    shape = torch.Size([len(sparse_mxs), sparse_mxs[0].shape[0], max_shape, max_shape])
    return torch.sparse.FloatTensor(indices, values, shape)