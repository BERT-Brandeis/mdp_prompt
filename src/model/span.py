import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from transformers import AutoModel

from data.data_structures_modal import *
from model.shared import MLP


class MentionExtractor(nn.Module):
    def __init__(self, config, tag_to_ix):
        super(MentionExtractor, self).__init__()

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.bio_tag_scorer = MLP(config.hidden_size, self.tagset_size)
        self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, feats, input_ids, gold_tags, attention_mask=None):
        # feats: batch_size * max_seq_len * hidden
        scores = self.bio_tag_scorer(feats)
        scores = scores.view(-1, self.tagset_size)
        gold_tags = torch.squeeze(gold_tags.view(scores.shape[0], -1))
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = scores[active_loss]
            active_labels = gold_tags[active_loss]
            loss = self.loss_fct(active_logits, active_labels)
        else:
            loss = self.loss_fct(scores, gold_tags)
        return loss

    def forward(self, feats):
        scores = self.bio_tag_scorer(feats)
        _, tags = scores.max(dim=-1)
        return scores, tags


class SpanBasedModel(nn.Module):
    def __init__(self, config):
        super(SpanBasedModel, self).__init__()
        self.bert_dim = config.hidden_size
        self.bert = AutoModel.from_config(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.bio_extractor = MentionExtractor(config, tag_to_ix=PARSING_BIO2id) #PARSING_BIO2id
        self.rel_classifier = MentionExtractor(config, tag_to_ix=MODAL_EDGE_LABEL_LIST_SPAN_label2id)

    def load_bert(self, model_name, cache_dir=None):
        print('Loading pre-trained BERT model {}'.format(model_name))
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def encode(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_states = outputs[0] #.last_hidden_state
        last_hidden_states = self.bert_dropout(last_hidden_states)
        return last_hidden_states

    def forward(self, inputs):
        sequence_output = self.encode(inputs)
        loss_bio = self.bio_extractor.compute_loss(feats=sequence_output, input_ids=inputs['input_ids'],
                                                   gold_tags=inputs["input_bio_gold"],
                                                   attention_mask=inputs['attention_mask'])
        loss_rel = self.rel_classifier.compute_loss(feats=sequence_output, input_ids=inputs['input_ids'],
                                                    gold_tags=inputs["input_rel_gold"],
                                                    attention_mask=inputs['attention_mask'])
        return loss_bio + loss_rel

    def decode(self, inputs):
        sequence_output = self.encode(inputs)
        _, bio_tags = self.bio_extractor(sequence_output)
        _, rel_labels = self.rel_classifier(sequence_output)
        return bio_tags.detach().cpu().tolist(), rel_labels.detach().cpu().tolist()
