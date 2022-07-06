import numpy as np


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _flatten(lst):
    """Convert a list of list to list."""
    flatten_lst = []
    for upper_lst in lst:
        for item in upper_lst:
            flatten_lst.append(item)
    return flatten_lst


def _chunk_with_doc_stride(sequence, max_seq_len, doc_stride):
    """
    sequence: input ids without cls and sep; max_seq_len: max_seq_len - 2;
    """
    if len(sequence) <= max_seq_len:
        left_pad = [0] # cls
        right_pad = [0] # sep
        token_ratio = [[1] * len(sequence)]
        return [sequence], left_pad, right_pad, token_ratio

    all_seqs = []
    seq = []
    seq_starts = []
    for idx, token in enumerate(sequence):
        if len(seq) < max_seq_len - 1:
            seq.append(token)
            # the last token
            if idx == len(sequence) - 1:
                all_seqs.append(seq)
                seq_starts.append(idx + 1 - len(seq))
        elif len(seq) == max_seq_len - 1:
            seq.append(token)  # 510 token
            all_seqs.append(seq)
            seq_starts.append(idx + 1 - len(seq))
            # Start a new seq with doc_stride num tokens overlap
            if len(seq) <= doc_stride:
                print("not enough tokens...")
                print(len(seq), doc_stride)
            seq = seq[-doc_stride:]
            assert len(seq) == doc_stride

    left_pads = []
    right_pads = []
    token_ratio = []
    for s_id, seq in enumerate(all_seqs):
        if s_id == 0:
            assert len(seq) == max_seq_len
            left_pad = 0
            right_pad = len(sequence) - len(seq)
            assert right_pad >= 0
        else:
            seq_start = seq_starts[s_id]
            left_pad = seq_start
            right_pad = len(sequence) - seq_start - len(seq)
            assert left_pad > 0
            assert right_pad >= 0
        assert len(seq) + left_pad + right_pad == len(sequence)
        left_pads.append(left_pad)
        right_pads.append(right_pad)
        ratio = [0]*left_pad + [1]*len(seq) + [0]*right_pad
        token_ratio.append(ratio)

    assert len(all_seqs) == len(left_pads) == len(right_pads)
    return all_seqs, left_pads, right_pads, token_ratio


def tokenize_doc_with_overlap(one_doc, tokenizer, max_seq_len, is_split_into_words, doc_stride_ratio):
    """
    Tokenize one document into multiple chunks,
    each chunk has max_seq_len num of tokens.
    Each chunk overlaps previous chunk by doc_stride_ratio*max_seq_len tokens.
    Returns: a lisk of chunks, a list of token2subtoken_maps, a list of subtoken2token maps.
    """

    # Convert a list of sentences to a list of tokens
    one_doc = _flatten(one_doc.sentences_before_tokenization)

    tokenized_doc = tokenizer(
        one_doc,
        truncation=False,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        is_split_into_words=is_split_into_words,
        padding=False,
    )

    subtoken2token = tokenized_doc.word_ids()
    token2subtoken_map = {}
    previous_word = 'Null'
    for word in subtoken2token:
        if word is None:
            token2subtoken_map[word] = (None, None)
        elif word != previous_word:
            words = tokenized_doc.word_to_tokens(word)
            token2subtoken_map[word] = (words.start - 1, words.end - 1 - 1)
        previous_word = word

    input_id_chunks, left_pads, right_pads, token_ratio = _chunk_with_doc_stride(
        tokenized_doc['input_ids'][0][1:-1], max_seq_len - 2,
        doc_stride=int((max_seq_len - 2) * doc_stride_ratio))

    token_ratio = np.array(token_ratio)
    token_ratio = np.sum(token_ratio, axis=0).tolist()
    token_ratio = [1/t for t in token_ratio]

    input_id_chunks_updated, attention_mask_chunks_updated = [], []
    for i, input_id_chk in enumerate(input_id_chunks):
        input_id_chk = [tokenizer.cls_token_id] + input_id_chk
        input_id_chk += [tokenizer.sep_token_id]

        attention_mask_chk = [1] * len(input_id_chk)

        input_id_pad = [tokenizer.pad_token_id] * (max_seq_len - len(input_id_chk))
        atten_mask_pad = [0] * (max_seq_len - len(attention_mask_chk))
        input_id_chk += input_id_pad
        attention_mask_chk += atten_mask_pad
        assert len(input_id_chk) == len(attention_mask_chk) == max_seq_len

        input_id_chunks_updated.append(input_id_chk)
        attention_mask_chunks_updated.append(attention_mask_chk)

    overall_input_ids = tokenized_doc['input_ids'][0]

    return input_id_chunks_updated, left_pads, right_pads, token_ratio,\
           attention_mask_chunks_updated, token2subtoken_map, subtoken2token, overall_input_ids


def convert_dict2list_of_dict(dict):
    lst = []
    keys = dict.keys()

    for exm_id, exm in enumerate(dict["input_ids"]):
        one_dic = {}
        for k in keys:
            one_dic[k] = dict[k][exm_id]
        lst.append(one_dic)
    return lst


def tokenize_query_context(query, context, tokenizer, max_seq_len, is_split_into_words, doc_stride):
    tokenized_examples = tokenizer(
        query,
        context,
        truncation="only_second",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        is_split_into_words=True,
        padding="max_length")

    def get_token2subtoken_map(subtokens):
        previous_token = -1000
        token2subtoken_map = {}
        for subtoken, token in enumerate(subtokens):
            if token != previous_token:
                token_subtoken_start = subtoken
                # if token is not None:
                #     print(original_text.split(" ")[token])
            token_subtoken_end = subtoken
            previous_token = token
            token2subtoken_map[token] = (token_subtoken_start, token_subtoken_end)
        return token2subtoken_map

    if tokenizer.cls_token == '<s>':
        context_offset = [-1, -1]
    else:
        assert tokenizer.cls_token == '[CLS]'
        context_offset = [-1]

    tokenized_examples["id_before_tokenization"] = []
    tokenized_examples["context_start_end"] = []
    tokenized_examples["token2subtoken_maps"] = []
    tokenized_examples["tokens"] = []
    for tokenized_exm_id, tokenized_exm in enumerate(tokenized_examples["input_ids"]):
        subtoken2token_list = tokenized_examples.word_ids(batch_index=tokenized_exm_id)

        context_start = None
        for id, item in enumerate(subtoken2token_list):
            if id != 0 and item is None:
                # This is SEP
                context_start = id + 1
                break
        if len(context_offset) == 2:
            context_start += 1

        # Split query and context subtoken2token list
        context_subtoken2token_list = subtoken2token_list[context_start:]
        context_wtho_paddings = [t for t in context_subtoken2token_list if t is not None]
        context_subtoken2token_list = context_subtoken2token_list[:len(context_wtho_paddings)]

        context_token2subtoken_map = get_token2subtoken_map(context_subtoken2token_list)
        tokenized_examples["context_start_end"].append((context_start, context_start + len(context_wtho_paddings)))
        tokenized_examples["token2subtoken_maps"].append(context_token2subtoken_map)
        tokenized_examples["tokens"].append(context_subtoken2token_list[:len(context_wtho_paddings)])
        assert tokenized_examples["tokens"][-1] == context_subtoken2token_list

        tokenized_examples["id_before_tokenization"].append(tokenized_examples["overflow_to_sample_mapping"][tokenized_exm_id])

    tokenized_examples = convert_dict2list_of_dict(tokenized_examples)
    return tokenized_examples
