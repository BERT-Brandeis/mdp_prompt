import argparse
import logging
import os
import random
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

from model.ranking import PipeLineStage2, End2End, PipeLineStage1
from model.span import SpanBasedModel
from data.reader_doc import read_data, convert_to_batch
from data.reader_doc_qa import read_data as read_data_span
from data.reader_doc_qa import convert_to_batch as convert_to_batch_span
from data.data_structures_modal import MODAL_EDGE_LABEL_LIST, TEMPORAL_EDGE_LABEL_LIST, EVENT_CONC_BIO2id
from eval_dev import parse_pipeline_stage2, parse_end2end, parse_pipeline_stage1
from eval_dev import parse_span
from params import get_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    Config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    assert args.data_type is not None
    assert args.language is not None
    assert args.output_dir is not None
    output_dir = args.output_dir

    if args.classifier == 'pipeline_stage1':
        assert args.extract_event or args.extract_conc

    if args.data_type == 'modal':
        EDGE_LABEL_LIST = MODAL_EDGE_LABEL_LIST
    else:
        EDGE_LABEL_LIST = TEMPORAL_EDGE_LABEL_LIST

    params = get_parameters(args)

    if args.classifier == 'pipeline_stage2':
        model = PipeLineStage2(config=Config, params=params, edge_labels=EDGE_LABEL_LIST)
    elif args.classifier == 'pipeline_stage1':
        model = PipeLineStage1(config=Config, params=params)
    elif args.classifier == 'end2end':
        model = End2End(config=Config, params=params, edge_labels=EDGE_LABEL_LIST)
    elif args.classifier == 'span':
        model = SpanBasedModel(config=Config)

    if args.outmodel_name:
        state = torch.load(os.path.join(output_dir, args.outmodel_name + '_' + "pytorch_model.bin"))
        model.load_state_dict(state['model'])
    else:
        state = torch.load(os.path.join(output_dir, "pytorch_model.bin"))
        model.load_state_dict(state)
    model.to(device)

    eval_log_name = args.model + '_' + args.classifier + '_' + "parse.log"
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, eval_log_name), 'w'))
    logger.info(args)

    if args.input_file is not None:
        if args.classifier != "span":
            test_examples, test_lm_inputs, test_gcn_inputs = read_data(
                args, tokenizer, input_file=args.input_file, is_training=False,
                data_type=args.data_type, language=args.language)
            test_features = convert_to_batch(one_doc_example_input=test_examples,
                                             pretrained_lm_input=test_lm_inputs,
                                             gcn_input=test_gcn_inputs,
                                             device=device)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(test_examples))
            logger.info("  Num split examples = %d", len(test_features))
        else:
            test_examples, test_lm_inputs = read_data_span(
                args, tokenizer, input_file=args.input_file, is_training=False,
                data_type=args.data_type, language=args.language)
            test_features = convert_to_batch_span(doc_example_input=test_examples,
                                                  tokenized_example_lst=test_lm_inputs,
                                                  batch_size=args.eval_batch_size,
                                                  max_seq_len=args.max_seq_length,
                                                  device=device)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(test_lm_inputs))
            logger.info("  Num split examples = %d", len(test_features))
    else:
        assert args.input_plain is not None

    if args.parse_stage1:
        assert args.input_plain is not None
    if args.input_plain is None:
        test_plain_examples, test_plain_features = test_examples, test_features
    else:
        if args.classifier != "span":
            test_plain_examples, test_plain_lm_inputs, test_plain_gcn_inputs = read_data(
                args, tokenizer, input_file=args.input_plain, is_training=False,
                data_type=args.data_type, language=args.language)
            test_plain_features = convert_to_batch(one_doc_example_input=test_plain_examples,
                                                        pretrained_lm_input=test_plain_lm_inputs,
                                                        gcn_input=test_plain_gcn_inputs,
                                                        device=device)
            logger.info("***** Test plain *****")
            logger.info("  Num orig examples = %d", len(test_plain_examples))
            logger.info("  Num split examples = %d", len(test_plain_features))
        else:
            test_plain_examples, test_plain_lm_inputs = read_data_span(
                args, tokenizer, input_file=args.input_plain, is_training=False,
                data_type=args.data_type, language=args.language)
            test_plain_features = convert_to_batch_span(doc_example_input=test_plain_examples,
                                                  tokenized_example_lst=test_plain_lm_inputs,
                                                  batch_size=args.eval_batch_size,
                                                  max_seq_len=args.max_seq_length,
                                                  device=device)
            logger.info("***** Test plain *****")
            logger.info("  Num orig examples = %d", len(test_plain_lm_inputs))
            logger.info("  Num split examples = %d", len(test_plain_features))

    # if args.parse_stage1 and "test" in args.input_plain:
    #     if args.input_file:
    #         assert "test" in args.input_file
    #         assert "dev" not in args.input_file
    #     assert "dev" not in args.input_plain
    #     test_suffix = '_test_' + args.outfile_name
    # elif args.parse_stage1 and "dev" in args.input_plain:
    #     if args.input_file:
    #         assert "dev" in args.input_file
    #         assert "test" not in args.input_file
    #     assert "test" not in args.input_plain
    #     test_suffix = '_dev_' + args.outfile_name
    # else:
    test_suffix = "_preds"
    # elif args.input_file and "test" in args.input_file:
    #     # assert "dev" not in args.input_file
    #     test_suffix = '_test_' + args.outfile_name
    # elif args.input_file and "dev" in args.input_file:
    #     # assert "test" not in args.input_file
    #     test_suffix = '_dev_' + args.outfile_name
    # elif args.input_file and "test" in args.input_plain:
    #     # assert "dev" not in args.input_plain
    #     test_suffix = '_test_' + args.outfile_name
    # elif args.input_file and "dev" in args.input_plain:
    #     # assert "test" not in args.input_plain
    #     test_suffix = '_dev_' + args.outfile_name

    if args.classifier == "end2end":
        if args.parse_stage1:
            test_stage1_string = parse_end2end(
                model, eval_features=test_plain_features, data_type=args.data_type, stage1_only=True)
            with open(os.path.join(output_dir, str(args.model) + test_suffix + '_stage1.txt'),
                      'w', encoding='utf8') as ft1:
                ft1.write(test_stage1_string)
        if args.parse_stage2:
            # Make sure the input is parsed stage1 output
            assert 'stage1' in args.input_file
            test_stage2_string = parse_end2end(
                model, eval_features=test_features, data_type=args.data_type, stage2_only=True)
            with open(os.path.join(output_dir, str(args.model) + test_suffix + '_auto_nodes.txt'),
                      'w', encoding='utf8') as ft2:
                ft2.write(test_stage2_string)
    elif args.classifier == "pipeline_stage1":
        assert args.input_plain is not None
        test_stage1_string = parse_pipeline_stage1(
            model, eval_features=test_plain_features, data_type=args.data_type)
        with open(os.path.join(output_dir, str(args.model) + '_stage1.txt'),
                  'w', encoding='utf8') as ft1:
            ft1.write(test_stage1_string)
    elif args.classifier == "pipeline_stage2":
        if args.parse_stage2_gold:
            test_stage2_string = parse_pipeline_stage2(
                model, eval_features=test_features, data_type=args.data_type)
            with open(os.path.join(output_dir, str(args.model) + '_stage2.txt'),
                      'w', encoding='utf8') as ft1:
                ft1.write(test_stage2_string)
    elif args.classifier == 'span':
        test_stage2_string = parse_span(model, eval_features=test_features)
        with open(os.path.join(output_dir, str(args.model) + test_suffix + '_auto_nodes.txt'),
                  'w', encoding='utf8') as ft1:
            ft1.write(test_stage2_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None, type=str, required=True)

    parser.add_argument("--input_file", default=None, type=str, required=False)
    parser.add_argument("--input_plain", default=None, type=str, required=False)

    parser.add_argument("--data_type", default=None, type=str,
                        choices=['modal', 'temporal_event', 'temporal_time', 'temporal_tdt'], required=True)
    parser.add_argument("--language", default=None, type=str,
                        choices=['chn', 'eng'], required=True)

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--encoding_method", default="overlap", type=str, choices=["no_overlap", "overlap"])

    parser.add_argument("--parse_stage1", action='store_true')
    parser.add_argument("--parse_stage2_gold", action='store_true')
    parser.add_argument("--parse_stage2", action='store_true')

    parser.add_argument("--classifier", default=None, type=str,
                        choices=['pipeline_stage2', 'pipeline_stage1', 'multi_task', 'end2end', 'span'], required=True)
    parser.add_argument("--num_labels", default=None, type=int, required=True)

    parser.add_argument("--use_rgcn", action='store_true', default=False)
    parser.add_argument("--rgcn_type", default=None, type=str, choices=["rgcn", "rgcn_aggr"])
    parser.add_argument("--rgcn_relation_cnt", default=3, type=int, required=False)
    parser.add_argument('--dist_dim', type=int, default=0)

    parser.add_argument("--extract_event", action='store_true', default=False)
    parser.add_argument("--extract_conc", action='store_true', default=False)

    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--outfile_name", default=None, type=str)
    parser.add_argument("--outmodel_name", default=None, type=str, required=False)

    parser.add_argument("--eval_batch_size", default=16, type=int)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    main(args)

