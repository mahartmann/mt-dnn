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


def rejoin_subwords(toks, labels):
    filtered_toks = []
    filtered_labels = []
    for i, tok in enumerate(toks):
        label = labels[i]
        if tok.startswith('##'):
            filtered_toks[-1].append(tok.strip('##'))
        else:
            filtered_toks.append([tok])
            filtered_labels.append(label)
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder_type = config.get('encoder_type', EncoderModelType.BERT)

    test_data_set = SingleTaskDataset(args.prep_input, False, maxlen=args.max_seq_len, task_id=args.task_id, task_def=task_def)
    collater = Collater(is_train=False, encoder_type=encoder_type)
    test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=collater.collate_fn, pin_memory=args.cuda)

    with torch.no_grad():
        test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data,
                                                                             metric_meta=metric_meta,
                                                                             use_cuda=args.cuda, with_label=args.with_label)

        results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
        uid2pred = {}
        for uid, pred in zip(results['uids'], results['predictions']):
            uid2pred[uid] = pred
        silver_signal = 'scope'
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



                filtered_toks, filtered_labels = rejoin_subwords(filtered_toks, filtered_labels)
                assert len(filtered_toks) == len(filtered_labels)
                out_labels.append(filtered_labels)
                out_seqs.append(filtered_toks)


            if silver_signal == 'cue':
                data = []
                # produce cue annotated data
                for seq, labels in zip(out_seqs, out_labels):
                    print(seq)
                    cue_labelseq = [0 if label_map[elm] != 'I' else 1 for elm in labels]
                    scope_labels = [None for elm in seq]
                    assert len(scope_labels) == len(cue_labelseq) == len(seq)
                    if setting == 'embed':
                        data.append([scope_labels, seq, cue_labelseq])
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
                        data.append([augmented_labels, augmented_seq, augmented_cue_labelseq])
                write_split(fname='silver_cues_augmented.tsv', data=[[i] + elm for i, elm in enumerate(data)])


            elif silver_signal == 'scope':
                data = []
                # produce scope annotated data
                for seq, labels in zip(out_seqs, out_labels):
                    print(seq)
                    cue_labelseq = [None for elm in seq]
                    scope_labels = [elm for elm in labels]
                    assert len(scope_labels) == len(cue_labelseq) == len(seq)

                    data.append([scope_labels, seq, cue_labelseq])
                write_split(fname='silver_scope.tsv', data=data)



        #dump(args.score, results)
        #if args.with_label:
        #    print(test_metrics)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_def", type=str, default="experiments/negscope/cue_task_def.yml")
    parser.add_argument("--task", type=str, default='sherlocken#cues')
    parser.add_argument("--task_id", type=int, help="the id of this task when training")

    parser.add_argument("--prep_input", type=str,
                        default="/home/mareike/PycharmProjects/negScope/data/formatted/bert-base-uncased_lower/sherlocken#cues_train.json")
    parser.add_argument("--with_label", action="store_true")
    parser.add_argument("--score", type=str, help="score output path", default='tmp')
    #parser.add_argument("--tokenizer", type=str, help='tokenizer for processing the raw input', 'default')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    parser.add_argument("--checkpoint", default='checkpoint/f1b35a39-4615-4b02-9718-02c6179c63da/model_0.pt', type=str)
    #parser.add_argument("--checkpoint", default='checkpoint/f1b35a39-4615-4b02-9718-02c6179c63da/model_0.pt', type=str)
    args = parser.parse_args()
    main(args)