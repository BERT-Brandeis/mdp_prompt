import argparse
import logging
import os
import random
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from model.ranking import PipeLineStage2, End2End, PipeLineStage1
from model.span import SpanBasedModel
from data.reader_doc import read_data, convert_to_batch
from data.reader_doc_qa import read_data as read_data_span
from data.reader_doc_qa import convert_to_batch as convert_to_batch_span
from data.data_structures_modal import MODAL_EDGE_LABEL_LIST, TEMPORAL_EDGE_LABEL_LIST, EVENT_CONC_BIO2id
from eval_dev import parse_and_eval_pipeline_stage2, parse_and_eval_end2end, parse_and_eval_pipeline_stage1
from eval_dev import parse_and_eval_span
from params import get_parameters


try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    if not os.path.exists(args.best_on_dev_output_dir):
        os.makedirs(args.best_on_dev_output_dir)
    best_on_dev_output_dir = args.best_on_dev_output_dir

    train_log_name = args.model + '_' + args.classifier + '_' + args.outmodel_name + '_' + "train.log"
    logger.addHandler(logging.FileHandler(os.path.join(args.best_on_dev_output_dir, train_log_name), 'w'))
    logger.info(args)

    Config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    assert args.data_type is not None
    assert args.language is not None

    if args.classifier == 'pipeline_stage1':
        assert args.extract_event or args.extract_conc

    if args.classifier != "span":
        train_examples, train_lm_inputs, train_gcn_inputs = read_data(
            args, tokenizer, input_file=args.train_file, is_training=True,
            data_type=args.data_type, language=args.language)
        train_features = convert_to_batch(one_doc_example_input=train_examples,
                                          pretrained_lm_input=train_lm_inputs,
                                          gcn_input=train_gcn_inputs,
                                          device=device)
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
    else:
        train_examples, train_lm_inputs = read_data_span(
            args, tokenizer, input_file=args.train_file, is_training=True,
            data_type=args.data_type, language=args.language)
        train_features = convert_to_batch_span(doc_example_input=train_examples,
                                          tokenized_example_lst=train_lm_inputs,
                                          batch_size=args.train_batch_size,
                                          max_seq_len=args.max_seq_length,
                                          device=device)
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_lm_inputs))
        logger.info("  Num split examples = %d", len(train_features))

    if args.mix_data:
        assert not args.pretraining
    if args.pretraining:
        assert not args.mix_data

    if args.mix_data or args.pretraining:
        assert args.train_file_pretrain is not None
        tr_pretrain_examples, tr_pretrain_lm_inputs, tr_pretrain_gcn_inputs = read_data(
            args, tokenizer, input_file=args.train_file_pretrain, is_training=True,
            data_type=args.data_type, language=args.language)
        tr_pretrain_features = convert_to_batch(one_doc_example_input=tr_pretrain_examples,
                                                      pretrained_lm_input=tr_pretrain_lm_inputs,
                                                      gcn_input=tr_pretrain_gcn_inputs,
                                                      device=device)
        logger.info("***** Pre Train *****")
        logger.info("  Num orig examples = %d", len(tr_pretrain_examples))
        logger.info("  Num split examples = %d", len(tr_pretrain_features))

        if args.mix_data:
            train_features += tr_pretrain_features

            logger.info("***** Mix Train and Pre Train in Total *****")
            logger.info("  Num orig examples = %d", len(train_features))
            logger.info("  Num split examples = %d", len(train_features))

    num_train_optimization_steps = len(train_features) * args.num_train_epochs
    logger.info("  Num steps = %d", num_train_optimization_steps)

    if args.classifier != "span":
        dev_examples, dev_lm_inputs, dev_gcn_inputs = read_data(
            args, tokenizer, input_file=args.dev_file, is_training=False,
            data_type=args.data_type, language=args.language)
        dev_features = convert_to_batch(one_doc_example_input=dev_examples,
                                        pretrained_lm_input=dev_lm_inputs,
                                        gcn_input=dev_gcn_inputs,
                                        device=device)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(dev_examples))
        logger.info("  Num split examples = %d", len(dev_features))
    else:
        dev_examples, dev_lm_inputs = read_data_span(
            args, tokenizer, input_file=args.dev_file, is_training=False,
            data_type=args.data_type, language=args.language)
        dev_features = convert_to_batch_span(doc_example_input=dev_examples,
                                          tokenized_example_lst=dev_lm_inputs,
                                          batch_size=args.train_batch_size,
                                          max_seq_len=args.max_seq_length,
                                          device=device)
        dev_eval_examples, dev_eval_lm_inputs = read_data_span(
            args, tokenizer, input_file=args.dev_eval_file, is_training=False,
            data_type=args.data_type, language=args.language)
        dev_eval_features = convert_to_batch_span(doc_example_input=dev_eval_examples,
                                          tokenized_example_lst=dev_eval_lm_inputs,
                                          batch_size=args.train_batch_size,
                                          max_seq_len=args.max_seq_length,
                                          device=device)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(dev_lm_inputs))
        logger.info("  Num split examples = %d", len(dev_features))

    if args.mix_data:
        eval_per_epoch = args.eval_per_epoch * 2
    else:
        eval_per_epoch = args.eval_per_epoch
    eval_step = max(1, len(train_features) // eval_per_epoch)

    if args.data_type == 'modal':
        EDGE_LABEL_LIST = MODAL_EDGE_LABEL_LIST
    else:
        EDGE_LABEL_LIST = TEMPORAL_EDGE_LABEL_LIST

    params = get_parameters(args)
    #if not args.use_rgcn:
    #    assert params["dist_dim"] == 0

    if args.classifier == 'pipeline_stage1':
        model = PipeLineStage1(config=Config, params=params)
    elif args.classifier == 'pipeline_stage2':
        model = PipeLineStage2(config=Config, params=params, edge_labels=EDGE_LABEL_LIST)
    elif args.classifier == 'end2end':
        model = End2End(config=Config, params=params, edge_labels=EDGE_LABEL_LIST)
    elif args.classifier == 'span':
        model = SpanBasedModel(config=Config)

    model.load_bert(model_name=args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    model.to(device)

    if n_gpu > 1:
        if args.distributed_training:
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://localhost:23456',
                                                 rank=0, world_size=1)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    # from prettytable import PrettyTable
    # table = PrettyTable(["Modules", "Parameters"])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     param = parameter.numel()
    #     table.add_row([name, param])
    #     total_params += param

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)
    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=num_train_optimization_steps * args.warmup_proportion,
                                               num_training_steps=num_train_optimization_steps)

    tr_loss = 0
    dev_loss_inc_count = 0
    previous_dev_micro_f = -1
    previous_dev_micro_f_eval = -1

    nb_tr_examples = 0
    nb_tr_steps = 0
    global_step = 0

    pred_nodes_ratio = 0
    start_time = time.time()
    # assert args.eval_on_dev

    for epoch in range(int(args.num_train_epochs)):
        model.train()

        logger.info("Start epoch #{} (lr = {})...".format(epoch, args.learning_rate))

        random.shuffle(train_features)
        if not args.pretraining:
            assert args.pretrain_epochs < 1
            training_features_to_use = train_features
        else:
            assert args.pretrain_epochs >= 1
            random.shuffle(tr_pretrain_features)
            if epoch <= args.pretrain_epochs:
                training_features_to_use = tr_pretrain_features
            else:
                training_features_to_use = train_features

        if args.classifier == 'end2end' and epoch >= 2:
            use_pred_edges = True
            pred_nodes_ratio += 0.1
            pred_nodes_ratio = min(1., pred_nodes_ratio)
        else:
            use_pred_edges = False

        for step, train_feature in enumerate(training_features_to_use):
            if args.classifier in ['end2end', 'multi_task']:
                loss, loss_extraction, loss_parsing = model(
                   one_doc_example=train_feature["doc_example"],
                   lm_input=train_feature["pretrained_lm_input"],
                   gcn_input=train_feature["gcn_input"],
                   pred_nodes_ratio=pred_nodes_ratio,
                   is_training=True,
                   use_pred_edges=use_pred_edges)
            elif args.classifier == 'pipeline_stage1':
                loss = model(one_doc_example=train_feature["doc_example"],
                            lm_input=train_feature["pretrained_lm_input"],
                            is_training=True)
            elif args.classifier == 'span':
                loss = model(train_feature)
            else:
                loss = model(one_doc_example=train_feature["doc_example"],
                            lm_input=train_feature["pretrained_lm_input"],
                            gcn_input=train_feature["gcn_input"],
                            is_training=True,
                            use_pred_edges=use_pred_edges)

            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            nb_tr_examples += 1
            nb_tr_steps += 1

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % eval_step == 0:
                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                   epoch, step + 1, len(training_features_to_use), time.time() - start_time, tr_loss / nb_tr_steps))

                save_best_on_dev_micro = False
                # save_best_on_eval_dev_micro = False
                if args.eval_on_dev:
                    dev_macro_f, dev_micro_f = -1, -1
                    dev_micro_f_eval = -1

                    if args.classifier == 'pipeline_stage2':
                        dev_macro_f, dev_micro_f = parse_and_eval_pipeline_stage2(
                           model, eval_features=dev_features, data_type=args.data_type, gold_file=args.dev_file)
                        logger.info("dev macro_f {:.3f}, dev micro_f {:.3f}".format(dev_macro_f, dev_micro_f))

                    elif args.classifier == 'span':
                        dev_macro_f, dev_micro_f = parse_and_eval_span(
                           model=model, eval_features=dev_features, data_type=args.data_type, gold_file=args.dev_file)
                        logger.info("dev macro_f {:.3f}, dev micro_f {:.3f}".format(dev_macro_f, dev_micro_f))

                        dev_macro_f_eval, dev_micro_f_eval = parse_and_eval_span(
                           model=model, eval_features=dev_eval_features, data_type=args.data_type,
                           gold_file=args.dev_file)
                        logger.info("dev eval macro_f {:.3f}, dev micro_f {:.3f}".format(dev_macro_f_eval, dev_micro_f_eval))

                    elif args.classifier == 'end2end':
                        e_macro_f, e_micro_f, conc_macro_f, conc_micro_f, macro_f, micro_f = parse_and_eval_end2end(
                           model, eval_features=dev_features, data_type=args.data_type, gold_file=args.dev_file)

                        dev_micro_f = micro_f #(conc_micro_f + micro_f) / 2
                        logger.info(
                           "dev event macro_f {:.3f}, dev event micro_f {:.3f}, "
                           "dev conc macro_f {:.3f}, dev conc micro_f {:.3f}, "
                           "dev macro f {:.3f}, dev micro f {:.3f}".format(
                               e_macro_f, e_micro_f, conc_macro_f, conc_micro_f, macro_f, micro_f))

                    elif args.classifier == "pipeline_stage1":
                        e_macro_f, e_micro_f, conc_macro_f, conc_micro_f = parse_and_eval_pipeline_stage1(
                           args, model, eval_features=dev_features, data_type=args.data_type, gold_file=args.dev_file)
                        logger.info(
                           "dev event macro_f {:.3f}, dev event micro_f {:.3f}, "
                           "dev conc macro_f {:.3f}, dev conc micro_f {:.3f}".format(
                               e_macro_f, e_micro_f, conc_macro_f, conc_micro_f))
                        dev_micro_f = e_micro_f + conc_micro_f

                    if dev_micro_f >= previous_dev_micro_f:
                        previous_dev_micro_f = dev_micro_f
                        save_best_on_dev_micro = True
                        dev_loss_inc_count = 0
                    else:
                        if epoch >= args.early_stopping_warmup_threshold:
                            dev_loss_inc_count += 1

                    if dev_micro_f_eval >= previous_dev_micro_f_eval:
                        previous_dev_micro_f_eval = dev_micro_f_eval
                        save_best_on_eval_dev_micro = True

                    model.train()
                else:
                    save_best_on_dev_micro = True

                if save_best_on_dev_micro:
                    logger.info("save best micro model on dev at epoch %d step %d" % (epoch, step))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_on_dev_output_dir, args.outmodel_name + '_' + WEIGHTS_NAME )
                    state = dict(model=model_to_save.state_dict())
                    torch.save(state, output_model_file)
                    tokenizer.save_pretrained(best_on_dev_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)

    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--dev_file", default=None, type=str, required=True)
    parser.add_argument("--test_file", default=None, type=str, required=False)

    parser.add_argument("--dev_eval_file", default=None, type=str, required=False)
    parser.add_argument("--test_eval_file", default=None, type=str, required=False)

    parser.add_argument("--mix_data", action='store_true', help="Whether to mix two training data files.")
    parser.add_argument("--pretraining", action='store_true', help="Whether to pretrain on a larger dataset.")
    parser.add_argument("--train_file_pretrain", default=None, type=str, required=False)
    parser.add_argument("--dev_file_pretrain", default=None, type=str, required=False)
    parser.add_argument("--test_file_pretrain", default=None, type=str, required=False)

    parser.add_argument("--data_type", default=None, type=str,
                        choices=['modal', 'temporal_event', 'temporal_time', 'temporal_tdt'], required=True)
    parser.add_argument("--language", default=None, type=str, choices=['chn', 'eng'], required=True)

    parser.add_argument("--max_seq_length", default=384, type=int)
    parser.add_argument("--encoding_method", default="overlap", type=str, choices=["no_overlap", "overlap"])

    parser.add_argument("--classifier", default=None, type=str,
                        choices=['pipeline_stage2', 'pipeline_stage1', 'multi_task', 'end2end', 'span'], required=True)
    parser.add_argument("--num_labels", default=None, type=int, required=True)

    parser.add_argument("--use_rgcn", action='store_true', default=False)
    parser.add_argument("--rgcn_type", default=None, type=str, choices=["rgcn", "rgcn_aggr"])
    parser.add_argument("--rgcn_relation_cnt", default=3, type=int, required=False)

    parser.add_argument("--extract_event", action='store_true', default=False)
    parser.add_argument("--extract_conc", action='store_true', default=False)

    parser.add_argument("--eval_on_dev", action='store_true', help='Whether to eval on dev set during training.')
    parser.add_argument("--early_stopping_warmup_threshold",
                        help="iterations before starting early stopping", default=10, type=int)

    parser.add_argument("--best_on_dev_output_dir", default=None, type=str)
    parser.add_argument("--outmodel_name", default=None, type=str)

    parser.add_argument("--distributed_training", action='store_true')
    parser.add_argument("--num_train_epochs", default=30., type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--pretrain_epochs", default=0., type=float,
                        help="Total number of pretraining epochs to perform.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--eval_per_epoch", default=1, type=int)

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dist_dim', type=int, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% ""of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()
    main(args)


