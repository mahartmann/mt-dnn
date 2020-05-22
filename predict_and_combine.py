import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from data_utils.metrics import calc_metrics
from mt_dnn.inference import eval_model
from preprocessing.annotation_reader import get_clue_annotated_data
from preprocessing.data_splits import write_split

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def rejoin_subwords(toks, labels, replace_labels, default_label):
    filtered_toks = []
    filtered_labels = []
    for i, tok in enumerate(toks):

        label = labels[i]
        if tok.startswith('##'):
            filtered_toks[-1].append(tok.strip('##'))
        else:
            filtered_toks.append([tok])
            filtered_labels.append(label)
    #filtered_labels = [l if l not in replace_labels else default_label for l in filtered_labels]
    return [''.join(elm) for elm in filtered_toks], filtered_labels

def main(args):
    """"
    predict unlabeled data and use predictions as silver labels, either silver cue labels for input to scope prediction, or silver scopes as input for downstream task
    read in the raw data, tokenize, predict, and re-tokenize. this way the output can be used for several different models using prepro_std
    """
    # load task info
    task = args.task
    task_defs = TaskDefs(args.task_def)
    #print(task_defs._task_type_map)
    assert args.task in task_defs._task_type_map
    assert args.task in task_defs._data_type_map
    assert args.task in task_defs._metric_meta_map
    prefix = task.split('_')[0]
    task_def = task_defs.get_task_def(prefix)
    data_type = task_defs._data_type_map[args.task]
    task_type = task_defs._task_type_map[args.task]
    metric_meta = task_defs._metric_meta_map[args.task]
    # load model
    checkpoint_path = args.checkpoint
    assert os.path.exists(checkpoint_path)
    if args.cuda:
        state_dict = torch.load(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    config = state_dict['config']
    config["cuda"] = args.cuda
    model = MTDNNModel(config, state_dict=state_dict)
    model.load(checkpoint_path)

    #tokenizer = BertTokenizer.from_pretrained(model.config['bert_model_type'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    encoder_type = config.get('encoder_type', EncoderModelType.BERT)

    test_data_set = SingleTaskDataset(args.prep_input, False, maxlen=args.max_seq_len, task_id=args.task_id, task_def=task_def)
    collater = Collater(is_train=False, encoder_type=encoder_type)
    test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=collater.collate_fn, pin_memory=args.cuda)
    # get the sids from the datafile directly
    with open(args.prep_input) as f:
        data_sids = [json.loads(line)['sid'] for line in  f]
        print(data_sids)
    with torch.no_grad():
        test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data,
                                                                             metric_meta=metric_meta,
                                                                             use_cuda=args.cuda, with_label=args.with_label)

        results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
        uid2pred = {}
        for uid, pred in zip(results['uids'], results['predictions']):
            uid2pred[uid] = pred

        setting= 'augment'
        out_seqs = []
        out_labels = []
        for data in test_data:
            label_map = data[0]['task_def']['label_vocab']
            all_input_ids = data[1][0]
            uids = data[0]['uids']
            for uid, input_ids in zip(uids, all_input_ids):
                pred = uid2pred[uid]
                input_ids = [elm.item() for elm in input_ids if elm.item() != tokenizer.pad_token_id]

                toks = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
                assert len(input_ids) == len(pred)
                filtered_toks = []
                filtered_labels = []
                for i, tok in enumerate(toks):
                    if tok not in tokenizer.special_tokens_map.values():
                        filtered_toks.append(tok)
                        filtered_labels.append(pred[i])


                replace_labels = set([label_map[l] for l in ['CLS', 'SEP', 'X']])
                default_label = label_map['O']
                filtered_toks, filtered_labels = rejoin_subwords(filtered_toks, filtered_labels, replace_labels, default_label)
                assert len(filtered_toks) == len(filtered_labels)
                out_labels.append(filtered_labels)
                out_seqs.append(filtered_toks)


        if args.silver_signal == 'cue':
            """
            predict cues. if several cues are detected in one sentence, write two sentences with one cue each
            """
            data = []
            # produce cue annotated data
            for sid, seq in enumerate(out_seqs):
                labels = out_labels[sid]
                multi_cue_labelseq = [0 if label_map[elm] != '1' else 1 for elm in labels]
                # if we have n unrelated cues, replicate the labelseq n times

                def replicate_labelseq(seq):
                    replicates = []
                    i = 0
                    repl = [0] * i
                    while True:
                        if i >= len(seq): break
                        if seq[i] == 1:
                            repl.append(1)
                            i += 1
                        elif len(repl) > 0 and seq[i] == 0 and repl[-1] == 1:
                            # fill repl and append
                            repl += [0] * (len(seq) - len(repl))
                            replicates.append(repl)
                            repl = [0] * i
                        else:
                            repl.append(seq[i])
                            i += 1
                    if 1 in repl:
                        replicates.append(repl)
                    #print(replicates)
                    return replicates

                for cid, cue_labelseq in enumerate(replicate_labelseq(multi_cue_labelseq)):
                    scope_labels = [None for elm in seq]

                    assert len(scope_labels) == len(cue_labelseq) == len(seq)
                    uid = len(data)

                    if setting == 'embed':
                        data.append({'uid':uid,
                                         'labels': scope_labels,
                                         'seq': seq,
                                         'cue_indicator':cue_labelseq,
                                         'sid':'{}_{}'.format(sid, cid)})
                    elif setting == 'augment':
                        augmented_cue_labelseq = []
                        augmented_labels = []
                        augmented_seq = []
                        for i, label in enumerate(cue_labelseq):
                            if label == 1:
                                augmented_seq.append('CUE')
                                augmented_cue_labelseq.append(label)
                                augmented_labels.append(None)
                            augmented_seq.append(seq[i])
                            augmented_cue_labelseq.append(label)
                            augmented_labels.append(None)
                        data.append({'uid': uid,
                                         'labels': augmented_labels,
                                         'seq': augmented_seq,
                                         'cue_indicator': augmented_cue_labelseq,
                                         'sid': '{}_{}'.format(sid, cid)})
            print(data)
            write_split(fname=args.outfile, data=data)


        elif args.silver_signal == 'scope':
            """
            predict scopes. if we make predictions for the same sentence, re-combine them into one sentence.
            """
            non_combined_data = []
            # produce scope annotated data
            for _seq, labels in zip(out_seqs, out_labels):
                seq = [elm for elm in _seq if elm !='CUE']
                scope_labels = [labels[i] for i,elm in enumerate(_seq) if elm != 'CUE']
                cue_labelseq = [None for elm in seq]

                assert len(scope_labels) == len(cue_labelseq) == len(seq)
                uid = len(non_combined_data)
                non_combined_data.append({'uid': uid,
                                 'labels': scope_labels,
                                 'seq': seq,
                                 'cue_indicator': cue_labelseq})
            combined_data = re_combine_data(data_sids, non_combined_data, label_mapper=label_map)
            print(combined_data)
            write_split(fname=args.outfile, data=combined_data)



        #dump(args.score, results)
        #if args.with_label:
        #    print(test_metrics)

def re_combine_data(sids, data, label_mapper):
    sid2data = {}
    for sid, elm in zip(sids, data):
        s = sid.split('_')[0]
        sid2data.setdefault(s, []).append(elm)
    combined_data = []
    seen_sids = set()
    keys = []
    for sid in sids:
        s = sid.split('_')[0]
        if s not in seen_sids:
            seen_sids.add(s)
            keys.append(s)
    for sid in keys:
        sents = sid2data[sid]

        seq = sents[0]['seq']
        uid = len(combined_data)
        labels = ['O']*len(seq)
        cue_indicator = [0]*len(seq)
        for sent in sents:
            for i, elm in enumerate(sent['labels']):
                if label_mapper[elm] == 'I':
                    labels[i] = 'I'
            for i, elm in enumerate(sent['cue_indicator']):
                if elm == 1:
                    cue_indicator[i] = 1
        print(labels)
        combined_data.append({'uid': uid,
                                 'labels': labels,
                                 'seq': seq,
                                 'cue_indicator': cue_indicator})

    return combined_data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_def", type=str, default="experiments/negscope/iula_task_def.yml")
    parser.add_argument("--task", type=str, default='iula#cues')
    parser.add_argument("--task_id", type=int, help="the id of this task when training")

    parser.add_argument("--prep_input", type=str,
                        default="/home/mareike/PycharmProjects/negscope/data/formatted/bert-base-cased/iulanocues_test.json")
    parser.add_argument("--outfile", type=str,
                        default="/home/mareike/PycharmProjects/negscope/data/formatted/iulasilvercue_test.tsv")
    parser.add_argument("--with_label", action="store_true")
    parser.add_argument("--score", type=str, help="score output path", default='tmp')
    parser.add_argument("--silver_signal", type=str, help='what we want to predict', choices=['scope', 'cue'], default='cue')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument("--checkpoint", default='checkpoint/scope/74ad6ba6-ced4-4649-9997-95338d2c8c9e/model_4.pt', type=str)
    args = parser.parse_args()
    main(args)