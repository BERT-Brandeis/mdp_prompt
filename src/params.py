from data.data_preparation_modal import eng_max_padded_candidate_length, chn_max_padded_candidate_length


def get_parameters(args):
    if args.language == "eng":
        num_candidate_parent = eng_max_padded_candidate_length
    else:
        num_candidate_parent = chn_max_padded_candidate_length

    params = {
        "use_rgcn": args.use_rgcn,
        "rgcn_type": args.rgcn_type,
        "concat_sent_vec": False,
        "stage1_dropout": 0.0,
        "gcn_in_drop": 0.0,
        "gcn_out_drop": 0.4,
        "rgcn_relation_cnt": args.rgcn_relation_cnt,
        "cuda": True,
        "type_dim": 0,
        "dist_dim": 0,
        "sent_vec": 'null', # gcn, orig or null
        "self_atten_head_num": 0,
        "decrease_node_dim_ratio": 0,
        "num_candidate_parent": num_candidate_parent
    }
    return params
