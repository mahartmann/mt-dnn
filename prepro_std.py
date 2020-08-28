# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat, get_enum_name_from_repr_str, get_additional_feature_names
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs, EncoderModelType
from experiments.squad import squad_utils
from pretrained_models import *
import configparser
from transformers import PreTrainedTokenizer


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mt_dnn_data_proc_{}.log'.format(MAX_SEQ_LEN))

def feature_extractor(tokenizer, text_a, text_b=None, max_length=512, model_type=None, enable_padding=False, pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=False,
                      additional_features_key=None): # set mask_padding_with_zero default value as False to keep consistent with original setting
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
    )

    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    if enable_padding:
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if model_type.lower() in ['bert', 'roberta']:
        attention_mask = None

    if model_type.lower() not in ['distilbert','bert', 'xlnet'] :
        token_type_ids = [0] * len(token_type_ids)

    return input_ids,attention_mask, token_type_ids # input_ids, input_mask, segment_id

def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly,
               max_seq_len=MAX_SEQ_LEN, encoderModelType=EncoderModelType.BERT, lab_dict=None, additional_features=None):
    def build_data_premise_only(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT, lab_dict=lab_dict):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = lab_dict[sample['label']]
                if len(premise) > max_seq_len - 2:
                    premise = premise[:max_seq_len - 2]
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, max_length=max_seq_len, model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))
    def build_data_premise_only_with_additional_features(data, dump_path, additional_features_ids, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT, lab_dict=lab_dict):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = lab_dict[sample['label']]
                tokens = []
                additional_features_extended = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):

                        if j == 0:
                            additional_features_extended.append({additional_features_id: sample[additional_features_id][i] for additional_features_id in additional_features_ids if type(sample[additional_features_id]) == list})
                        else:
                            # give all subwords the same additional feature as the first subword
                            additional_features_extended.append(
                                {additional_features_id: sample[additional_features_id][i] for additional_features_id in
                                 additional_features_ids if type(sample[additional_features_id]) == list})
                if len(tokens) > max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]
                    additional_features_extended = additional_features_extended[:max_seq_len - 2]
                # start and end symbol always get additiona feature 0
                additional_features = [{additional_features_id:0 for additional_features_id in additional_features_ids}] + additional_features_extended + [{additional_features_id:0 for additional_features_id in additional_features_ids}]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(additional_features) == len(input_ids)
                type_ids = [0] * len(input_ids)

                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids}
                for additional_features_id in additional_features_ids:
                    if additional_features_id == 'sid':
                        features['sid'] = additional_features[0]['sid']
                    else:
                        additional_features_seq = []
                        for elm in additional_features:
                            additional_features_seq.append(elm[additional_features_id])
                        features[additional_features_id] = additional_features_seq
                # convert the additional features to idxs
                for key in additional_features_ids:
                    if type(sample[key]) == list:
                        features[key] = [convert_additional_features(elm, key) for elm in features[key]]
                    else:
                        features[key] = sample[key]
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len,
                                                                    model_type=encoderModelType.name)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_list = sample['hypothesis']
                label = sample['label']
                input_ids_list = []
                type_ids_list = []
                for hypothesis in hypothesis_list:
                    input_ids, mask, type_ids = feature_extractor(tokenizer,
                                                                        premise, hypothesis, max_length=max_seq_len,
                                                                        model_type=encoderModelType.name)
                    input_ids_list.append(input_ids)
                    type_ids_list.append(type_ids)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids_list,
                    'type_id': type_ids_list,
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel']}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_sequence(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, encoderModelType=EncoderModelType.BERT, label_mapper=None):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                tokens = []
                labels = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):
                        if j == 0 or not subwords[j].startswith('##'):
                            labels.append(sample['label'][i])
                        else:
                            labels.append(label_mapper['X'])
                if len(tokens) > max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids)
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_sequence_with_additional_features(data, dump_path, additional_features_ids, max_seq_len=MAX_SEQ_LEN,
                                                     tokenizer=None,
                                                     encoderModelType=EncoderModelType.BERT, label_mapper=None):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                tokens = []
                labels = []
                additional_features = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):
                        if j == 0 or not subwords[j].startswith('##'):
                            labels.append(sample['label'][i])
                            additional_features.append({additional_features_id: sample[additional_features_id][i] for additional_features_id in additional_features_ids if additional_features_id!= 'sid'})
                        else:
                            labels.append(label_mapper['X'])
                            additional_features.append({additional_features_id: sample[additional_features_id][i] for additional_features_id in additional_features_ids if additional_features_id!= 'sid'})
                if len(tokens) > max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]
                    additional_features = additional_features[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                #additional_features = [0] + additional_features + [0]
                additional_features = [{additional_features_id: 0 for additional_features_id in
                 additional_features_ids if additional_features_id!= 'sid'}] + additional_features + [{additional_features_id: 0 for additional_features_id in
                 additional_features_ids if additional_features_id!= 'sid'}]

                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids) == len(additional_features)
                assert len(input_ids) <= max_seq_len
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}

                for additional_features_id in additional_features_ids:
                    if additional_features_id != 'sid':
                        features[additional_features_id] = [f[additional_features_id] for f in additional_features]
                    else:
                        features['sid'] = sample['sid']

                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_mrc(data, dump_path, max_seq_len=MRC_MAX_SEQ_LEN, tokenizer=None, label_mapper=None, is_training=True):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            unique_id = 1000000000 # TODO: this is from BERT, needed to remove it...
            for example_index, sample in enumerate(data):
                ids = sample['uid']
                doc = sample['premise']
                query = sample['hypothesis']
                label = sample['label']
                doc_tokens, cw_map = squad_utils.token_doc(doc)
                answer_start, answer_end, answer, is_impossible = squad_utils.parse_squad_label(label)
                answer_start_adjusted, answer_end_adjusted = squad_utils.recompute_span(answer, answer_start, cw_map)
                is_valid = squad_utils.is_valid_answer(doc_tokens, answer_start_adjusted, answer_end_adjusted, answer)
                if not is_valid: continue
                """
                TODO --xiaodl: support RoBERTa
                """
                feature_list = squad_utils.mrc_feature(tokenizer,
                                        unique_id,
                                        example_index,
                                        query,
                                        doc_tokens,
                                        answer_start_adjusted,
                                        answer_end_adjusted,
                                        is_impossible,
                                        max_seq_len,
                                        MAX_QUERY_LEN,
                                        DOC_STRIDE,
                                        answer_text=answer,
                                        is_training=True)
                unique_id += len(feature_list)
                for feature in feature_list:
                    so = json.dumps({'uid': ids,
                                'token_id' : feature.input_ids,
                                'mask': feature.input_mask,
                                'type_id': feature.segment_ids,
                                'example_index': feature.example_index,
                                'doc_span_index':feature.doc_span_index,
                                'tokens': feature.tokens,
                                'token_to_orig_map': feature.token_to_orig_map,
                                'token_is_max_context': feature.token_is_max_context,
                                'start_position': feature.start_position,
                                'end_position': feature.end_position,
                                'label': feature.is_impossible,
                                'doc': doc,
                                'doc_offset': feature.doc_offset,
                                'answer': [answer]})
                    writer.write('{}\n'.format(so))


    if data_format == DataFormat.PremiseOnly:
        if additional_features[0] != None:
            build_data_premise_only_with_additional_features(data, dump_path, additional_features, max_seq_len, tokenizer,
                                                         encoderModelType)
        else:
            build_data_premise_only(
                data,
                dump_path,
                max_seq_len,
                tokenizer,
                encoderModelType)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len, tokenizer, encoderModelType)
    elif data_format == DataFormat.Seqence:
        if additional_features:
            build_data_sequence_with_additional_features(data, dump_path, additional_features, max_seq_len, tokenizer,
                                                         encoderModelType, lab_dict)
        else:
            build_data_sequence(data, dump_path, max_seq_len, tokenizer, encoderModelType, lab_dict)
    elif data_format == DataFormat.MRC:
        build_data_mrc(data, dump_path, max_seq_len, tokenizer, encoderModelType)
    else:
        raise ValueError(data_format)

def convert_additional_features(feature, type='scope_indicator'):
    if type == 'scope_indicator':
        if feature == 'O':
            return 0
        elif feature == 'I':
            return 1
        elif feature == 0:
            return 0
        else: raise Exception
    else: return feature


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--model', type=str, default='bert-base-cased',
                        help='support all BERT, XLNET and ROBERTA family supported by HuggingFace Transformers', choices=['bert-base-multilingual-cased', 'spanish-bert-cased', 'bert-base-cased'])
    parser.add_argument('--literal_model_type', type=str, default='bert',
                        help='the type of base model, e.g. bert or xlnet')
    parser.add_argument('--json_format', type=bool_flag, default=True)
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--root_dir', type=str, default='/home/mareike/PycharmProjects/negscope/data/formatted/')
    parser.add_argument('--task_def', type=str, default="experiments/negscope/spbio_task_def.yml")
    parser.add_argument('--config', type=str, default='preprocessing/config.cfg')

    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    do_lower_case = args.do_lower_case
    root = args.root_dir
    assert os.path.exists(root)
    cfg = args.config
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    #literal_model_type = args.model.split('-')[0].upper()
    literal_model_type = args.literal_model_type.upper()
    encoder_model = EncoderModelType[literal_model_type]
    literal_model_type = literal_model_type.lower()
    mt_dnn_suffix = literal_model_type
    if 'base' in args.model:
        mt_dnn_suffix += "_base"
    elif 'large' in args.model:
        mt_dnn_suffix += "_large"

    config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_model_type]
    tokenizer = setup_customized_tokenizer(tokenizer_class=tokenizer_class, model=args.model, do_lower_case=do_lower_case, config=config)

    output_dir = args.model
    if 'uncased' in args.model:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = '{}_lower'.format(mt_dnn_suffix)
        output_dir = '{}_lower'.format(output_dir)

    mt_dnn_root = os.path.join(root, output_dir)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)

    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            file_path = os.path.join(root, "%s_%s.tsv" % (task, split_name))
            if not os.path.exists(file_path):
                logger.warning("File %s doesnot exit"%file_path)
                sys.exit(1)
            rows = load_data(file_path, task_def, json_format=args.json_format)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(
                rows,
                dump_path,
                tokenizer,
                task_def.data_type,
                encoderModelType=encoder_model,
                lab_dict=task_def.label_vocab,
                additional_features=get_additional_feature_names(task_def['additional_features']))


def setup_customized_tokenizer(model, tokenizer_class, do_lower_case, config):
    additional_tokens = []
    for i in range(10):
        additional_tokens.append('[START{}]'.format(i))
        additional_tokens.append('[END{}]'.format(i))
    additional_tokens.append('@CHEMICAL$')
    additional_tokens.append('@GENE$')
    additional_tokens.append('@DRUG$')
    additional_tokens.append('@DISEASE$')
    additional_tokens.append('[CUE]')
    if model == 'bert-base-multilingual-cased':
        tokenizer = tokenizer_class.from_pretrained(model, vocab_file=config.get('Files', 'mbertvocab'),
                                    do_lower_case=do_lower_case, additional_special_tokens=additional_tokens)
    if model == 'bert-base-cased':
        tokenizer = tokenizer_class.from_pretrained(model, vocab_file=config.get('Files', 'bertvocab'),
                                                    do_lower_case=do_lower_case,
                                                    additional_special_tokens=additional_tokens)
    elif model == 'spanish-bert-cased':
        #tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="dccuchile/bert-base-spanish-wwm-cased", vocab_file=config.get('Files', 'spanishbertvocab'),
        #                                            do_lower_case=do_lower_case,
        #                                            additional_special_tokens=additional_tokens)
        tokenizer = BertTokenizer(vocab_file=config.get('Files', 'spanishbertvocab'),
                                                   do_lower_case=do_lower_case,
                                                    additional_special_tokens=additional_tokens)
    return tokenizer

if __name__ == '__main__':
    args = parse_args()
    main(args)
