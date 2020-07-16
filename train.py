# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import uuid
from datetime import datetime
from pprint import pprint
import torch
from torch.utils.data import DataLoader

from my_utils import bool_flag
from pretrained_models import *
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model, extract_encoding
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler
from mt_dnn.model import MTDNNModel
import test_config


def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=5)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--annealed_sampling', type=float, default=0, help='Factor for annealed sampling, where tasks are sampled proportional to ds size towards beginning of training. If 0, sampling strategy is to draw proportional to ds size')
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    parser.add_argument('--num_hidden_layers', type=int, default=-1)
    parser.add_argument('--pretrained_model_config', type=str, default='')

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')

    #parser.add_argument("--init_checkpoint", default='/home/mareike/PycharmProjects/negscope/code/mt-dnn/checkpoint/nubes/best_model/model_best.pt', type=str)
    parser.add_argument("--init_checkpoint", default='test_config', type=str)
    parser.add_argument('--data_dir',
                        default='/home/mareike/PycharmProjects/negscope/data/formatted/bert-base-multilingual-cased')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/negscope/drugs_task_def.yml")
    parser.add_argument('--train_datasets', default='drugss,bio')
    parser.add_argument('--test_datasets', default='drugss')
    parser.add_argument('--load_intermed', default=False, type=bool_flag)
    parser.add_argument('--freeze', default=False, type=bool_flag, help ='freeze encoder layers')

    parser.add_argument('--glue_format_on', action='store_true')
    parser.add_argument('--mkd-opt', type=int, default=0, 
                        help=">0 to turn on knowledge distillation, requires 'softlabel' column in input data")

    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=10)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--save_best_only', type=bool_flag, default=True)
    parser.add_argument('--debug', type=bool_flag, default=False, help='enable debug mode')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    #parser.add_argument("--model_ckpt", default='/home/mareike/PycharmProjects/negscope/code/mt-dnn/checkpoint/nubes/best_model/model_best.pt', type=str)
    parser.add_argument("--model_ckpt",
                        default='test_config',
                        type=str)
    parser.add_argument("--resume", type=bool_flag, default=False)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint/drugs')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    #fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    return parser


parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
parser.add_argument('--encode_mode', action='store_true', help="only encode test data")

args = parser.parse_args()

output_dir = args.output_dir
if os.path.isdir(output_dir):
    # create new output dir
    output_dir = os.path.join(output_dir, str(uuid.uuid4()))
    logger.info('Specified output dir exists. Creating new output dir {}'.format(output_dir))
data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')
pprint(args)

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

if args.save_best_only:
    best_model_dir = os.path.join(output_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_dir = os.path.abspath(best_model_dir)
set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

task_defs = TaskDefs(args.task_def)
encoder_type = args.encoder_type

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data




def main():
    logger.info('Launching the MT-DNN training')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size

    tasks = {}
    task_def_list = []
    dropout_list = []

    train_datasets = []
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks: 
            continue
        task_id = len(tasks)
        tasks[prefix] = task_id
        task_def = task_defs.get_task_def(prefix)
        task_def_list.append(task_def)


        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        logger.info('Loading {} as task {}'.format(train_path, task_id))
        train_data_set = SingleTaskDataset(train_path, True, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
        train_datasets.append(train_data_set)
    train_collater = Collater(dropout_w=args.dropout_w, encoder_type=encoder_type, soft_label=args.mkd_opt > 0)
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio, annealed_sampling=args.annealed_sampling, max_epochs=args.epochs)
    multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, collate_fn=train_collater.collate_fn, pin_memory=args.cuda)

    opt['task_def_list'] = task_def_list

    dev_data_list = []
    test_data_list = []
    test_collater = Collater(is_train=False, encoder_type=encoder_type)
    for dataset in args.test_datasets:

        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        task_id = tasks[prefix]
        task_type = task_def.task_type
        data_type = task_def.data_type

        dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data_set = SingleTaskDataset(dev_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
            dev_data = DataLoader(dev_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data_set = SingleTaskDataset(test_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def)
            test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        test_data_list.append(test_data)

    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    # div number of grad accumulation. 
    num_all_batches = args.epochs * len(multi_task_train_data) // args.grad_accumulation_step
    logger.info('############# Gradient Accumulation Info #############')
    logger.info('number of step: {}'.format(args.epochs * len(multi_task_train_data)))
    logger.info('number of grad grad_accumulation step: {}'.format(args.grad_accumulation_step))
    logger.info('adjusted number of step: {}'.format(num_all_batches))
    logger.info('############# Gradient Accumulation Info #############')

    init_model = args.init_checkpoint
    state_dict = None

    if init_model == 'test_config':
        config = test_config.test_config
        args.load_intermed = False
    else:
        if os.path.exists(init_model):

            _state_dict = torch.load(init_model)

            if not 'config' in _state_dict:
                cfg = load_json(args.pretrained_model_config)
                state_dict = {'state':_state_dict, 'config':cfg}

            else:
                state_dict = _state_dict
            config = state_dict['config']
        else:
            if opt['encoder_type'] not in EncoderModelType._value2member_map_:
                raise ValueError("encoder_type is out of pre-defined types")
            literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()

            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            config = config_class.from_pretrained(init_model).to_dict()




    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    config['multi_gpu_on'] = opt["multi_gpu_on"]
    if args.num_hidden_layers != -1:
        config['num_hidden_layers'] = args.num_hidden_layers

    config['resume'] = args.resume
    config['model_ckpt'] = args.model_ckpt
    config['init_checkpoint'] = args.init_checkpoint
    config['load_intermed'] = args.load_intermed
    config['freeze'] = args.freeze
    config['train_datasets']  = args.train_datasets
    config['test_datasets'] = args.test_datasets
    config['output_dir'] = args.output_dir
    config['task_def'] = args.task_def
    config['task_def_list'] = opt['task_def_list']
    config['epochs']  = args.epochs
    config['learning_rate'] = args.learning_rate
    config['warmup'] = args.warmup




    opt.update(config)

    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)



    if not args.load_intermed and args.resume and args.model_ckpt:
        logger.info('loading model from {}'.format(args.model_ckpt))
        model.load(args.model_ckpt)

    if opt['freeze']:
        logger.info('freezing encoder')
        model._freeze_encoder()
    #### model meta str
    headline = '############# Model Arch of MT-DNN #############'
    ### print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))

    logger.info("Total number of params: {}".format(model.total_param))

    # tensorboard
    if args.tensorboard:
        args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
        tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)
    
    if args.encode_mode:
        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            test_data = test_data_list[idx]
            with torch.no_grad():
                encoding = extract_encoding(model, test_data, use_cuda=args.cuda)
            torch.save(encoding, os.path.join(output_dir, '{}_encoding.pt'.format(dataset)))
        return
    best_score = 0.0
    score = None
    dev_metric = None
    for n, p in model.network.named_parameters():
        if p.requires_grad:
            print(n)
    for epoch in range(0, args.epochs):
        logger.warning('At epoch {}'.format(epoch))
        start = datetime.now()

        for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):
            batch_meta, batch_data = Collater.patch_data(args.cuda, batch_meta, batch_data)
            task_id = batch_meta['task_id']
            model.update(batch_meta, batch_data)

            if (model.local_updates) % (args.log_per_updates * args.grad_accumulation_step) == 0 or model.local_updates == 1:
                ramaining_time = str((datetime.now() - start) / (i + 1) * (len(multi_task_train_data) - i - 1)).split('.')[0]
                logger.info('Task [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(task_id,
                                                                                                    model.updates,
                                                                                                    model.train_loss.avg,
                                                                                                    ramaining_time))
                if args.tensorboard:
                    tensorboard.add_scalar('train/loss', model.train_loss.avg, global_step=model.updates)


            if args.save_per_updates_on and ((model.local_updates) % (args.save_per_updates * args.grad_accumulation_step) == 0):
                model_file = os.path.join(output_dir, 'model_{}_{}.pt'.format(epoch, model.updates))
                logger.info('Saving mt-dnn model to {}'.format(model_file))
                model.save(model_file)

        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            task_def = task_defs.get_task_def(prefix)
            if idx == 0:
                dev_metric = task_def.metric_meta[0].name
            label_dict = task_def.label_vocab
            dev_data = dev_data_list[idx]
            if dev_data is not None:
                with torch.no_grad():
                    dev_metrics, dev_predictions, scores, golds, dev_ids= eval_model(model,
                                                                                    dev_data,
                                                                                     dataset=prefix,
                                                                                    metric_meta=task_def.metric_meta,
                                                                                    use_cuda=args.cuda,
                                                                                    label_mapper=label_dict,
                                                                                    task_type=task_def.task_type,
                                                                                    )
                for key, val in dev_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar('dev/{}/{}'.format(dataset, key), val, global_step=epoch)
                    if isinstance(val, str):
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    elif isinstance(val, dict):
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}:\n {3}'.format(dataset, epoch, key, val))
                    else:
                        logger.warning('Task {0} -- epoch {1} -- Dev {2}: {3:.3f}'.format(dataset, epoch, key, val))
                score_file = os.path.join(output_dir, '{}_dev_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': dev_metrics, 'predictions': dev_predictions, 'uids': dev_ids, 'scores': scores}
                print(results['metrics'])
                # track dev performance for the the first test task
                if idx == 0:
                    score = results['metrics'][dev_metric]
                dump(score_file, results)
                if args.glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir, '{}_dev_scores_{}.tsv'.format(dataset, epoch))
                    submit(official_score_file, results, label_dict)

            # test eval
            test_data = test_data_list[idx]
            if test_data is not None:
                with torch.no_grad():
                    test_metrics, test_predictions, scores, golds, test_ids= eval_model(model, test_data, dataset=prefix,
                                                                                        metric_meta=task_def.metric_meta,
                                                                                        use_cuda=args.cuda, with_label=True,
                                                                                        label_mapper=label_dict,
                                                                                        task_type=task_def.task_type)
                score_file = os.path.join(output_dir, '{}_test_scores_{}.json'.format(dataset, epoch))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
                dump(score_file, results)
                if args.glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir, '{}_test_scores_{}.tsv'.format(dataset, epoch))
                    submit(official_score_file, results, label_dict)
                logger.info('[new test scores saved.]')
        if args.save_best_only and score:
            if score > best_score:
                logger.info('new {} score {} > {}, saving model after epoch {} as best model to {}'.format(dev_metric, score, best_score, epoch, best_model_dir))
                best_score = score
                model_file = os.path.join(best_model_dir, 'model_best.pt')
                model.save(model_file)
        else:
            model_file = os.path.join(output_dir, 'model_{}.pt'.format(epoch))
            model.save(model_file)
    if args.tensorboard:
        tensorboard.close()


if __name__ == '__main__':
    main()


