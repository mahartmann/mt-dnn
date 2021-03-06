import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from data_utils.metrics import calc_metrics
from mt_dnn.inference import eval_model

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")

parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/negscope/scope_task_def.yml")
parser.add_argument("--task", type=str, default='biofull')
#parser.add_argument("--test_set", type=str, default='biofull#silvercues_train')
parser.add_argument("--task_id", type=int, help="the id of this task when training")

parser.add_argument("--prep_input", type=str,
                        default="/home/mareike/PycharmProjects/negscope/data/formatted/bert-base-cased/iula_test.json")
#parser.add_argument("--outfile", type=str,
#                        default="/home/mareike/PycharmProjects/negscope/data/formatted/biofull#silvercues_train.tsv")
parser.add_argument("--with_label", type=bool_flag, default=True)
parser.add_argument("--score", type=str, help="score output path", default='tmp')

parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--batch_size_eval', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
parser.add_argument("--checkpoint", default='/home/mareike/PycharmProjects/negscope/code/mt-dnn/checkpoint/nubes/best_model/model_best.pt', type=str)



args = parser.parse_args()

# load task info
task = args.task
task_defs = TaskDefs(args.task_def)
assert args.task in task_defs._task_type_map
assert args.task in task_defs._data_type_map
assert args.task in task_defs._metric_meta_map
prefix = task.split('_')[0]
task_def = task_defs.get_task_def(prefix)
data_type = task_defs._data_type_map[args.task]
task_type = task_defs._task_type_map[args.task]
metric_meta = task_defs._metric_meta_map[args.task]
for key in task_def.label_vocab:
    print(key)
print(task_def.label_vocab)
# load model
checkpoint_path = args.checkpoint
print(checkpoint_path)
assert os.path.exists(checkpoint_path)
if args.cuda:
    state_dict = torch.load(checkpoint_path)
else:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
config = state_dict['config']
config["cuda"] = args.cuda
model = MTDNNModel(config, state_dict=state_dict)
model.load(checkpoint_path)
encoder_type = config.get('encoder_type', EncoderModelType.BERT)
# load data
test_data_set = SingleTaskDataset(args.prep_input, False, maxlen=args.max_seq_len, task_id=args.task_id, task_def=task_def)
collater = Collater(is_train=False, encoder_type=encoder_type)
test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=collater.collate_fn, pin_memory=args.cuda)

with torch.no_grad():
    test_metrics, test_predictions, scores, golds, test_ids = eval_model(model, test_data, label_mapper=task_def.label_vocab,
                                                                         metric_meta=metric_meta,
                                                                         use_cuda=args.cuda, with_label=args.with_label,dataset=prefix)

    results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
    #print(results)
    dump(args.score, results)
    if args.with_label:
        print(test_metrics)



