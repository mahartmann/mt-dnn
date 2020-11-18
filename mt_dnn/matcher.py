# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import torch.nn as nn
from pretrained_models import MODEL_CLASSES
from transformers import BertConfig
from transformers import BertModel

from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader
from module.san_model import SanModel
from data_utils.task_def import EncoderModelType, TaskType, AdditionalFeatures
import tasks
from experiments.exp_def import TaskDef
from extensions.hooks import gradient_reversal_hook, MyHook

class LinearPooler(nn.Module):
    def __init__(self, hidden_size):
        super(LinearPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 3:
        opt_v = max_opt
    return opt_v

class SANBertNetwork(nn.Module):
    def __init__(self, opt,  bert_config=None, initial_from_local=False):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()

        self.forward_hooks = []
        self.backward_hooks = []
        self.scoring_forward_hooks = {}
        self.scoring_backward_hooks = {}

        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]

        self.preloaded_config = config_class.from_dict(opt)  # load config from opt
        self.bert = model_class(self.preloaded_config)
        hidden_size = self.bert.config.hidden_size

        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        task_def_list = opt['task_def_list']
        self.task_def_list = task_def_list
        self.decoder_opt = []
        self.task_types = []
        for task_id, task_def in enumerate(task_def_list):
            self.decoder_opt.append(generate_decoder_opt(task_def.enable_san, opt['answer_opt']))
            self.task_types.append(task_def.task_type)

        # create output header
        self.scoring_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        # create lists for additional inputs embeddings
        self.additional_input_features_list = nn.ModuleList()

        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            decoder_opt = self.decoder_opt[task_id]
            task_type = self.task_types[task_id]
            task_dropout_p = opt['dropout_p'] if task_def.dropout_p is None else task_def.dropout_p
            dropout = DropoutWrapper(task_dropout_p, opt['vb_dropout'])
            self.dropout_list.append(dropout)
            task_obj = tasks.get_task_obj(task_def)
            print('{}: {}'.format(task_id, task_obj))

            ################################################################
            ######## Add additional layers used during encoding ############
            ################################################################

            if task_def.additional_features is not None:
                for feature_name in task_def.additional_features:
                    if feature_name == AdditionalFeatures.cue_indicator:
                        embeds = nn.Embedding(2, hidden_size)
                        self.additional_input_features_list.append(embeds)
                    elif feature_name == AdditionalFeatures.scope_indicator:
                        embeds = nn.Embedding(2, hidden_size)
                        self.additional_input_features_list.append(embeds)
            else: self.additional_input_features_list.append(None)

            if task_obj is not None:
                print('Task obj for {} not None'.format(task_id))
                print('Setting out proj as with dec_opt {}, hid {}, lab {}'.format(decoder_opt,hidden_size,lab))
                out_proj = task_obj.train_build_task_layer(decoder_opt, hidden_size, lab, opt, prefix='answer', dropout=dropout)
            elif task_type == TaskType.Span:
                assert decoder_opt != 1
                out_proj = nn.Linear(hidden_size, 2)
            elif task_type == TaskType.SeqenceLabeling:
                out_proj = nn.Linear(hidden_size, lab)
            elif task_type == TaskType.MaskLM:
                if opt['encoder_type'] == EncoderModelType.ROBERTA:
                    # TODO: xiaodl
                    out_proj = MaskLmHeader(config=self.bert.config, embedding_weights=self.bert.embeddings.word_embeddings.weight)
                else:
                    out_proj = MaskLmHeader(config=self.bert.config, embedding_weights=self.bert.embeddings.word_embeddings.weight)
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(hidden_size, hidden_size, lab, opt, prefix='answer', dropout=dropout)
                else:
                    out_proj = nn.Linear(hidden_size, lab)
            if task_type == TaskType.Adversarial:
                # register the hook on the ouput layer
                out_proj.register_backward_hook(gradient_reversal_hook)
            # register hooks for checking gradients
            self.scoring_list.append(out_proj)
            self.scoring_forward_hooks[task_id] = (MyHook(out_proj))
            self.scoring_backward_hooks[task_id] = (MyHook(out_proj, backward=True))



        self.opt = opt
        self._my_init()

        # if not loading from local, loading model weights from pre-trained model, after initialization
        if not initial_from_local:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            #self.bert = model_class.from_pretrained(opt['init_checkpoint'],config=self.preloaded_config)
            self.bert = BertModel(self.preloaded_config)

        #register hooks
        for module in list(self.bert._modules.items()):
            self.forward_hooks.append((MyHook(module[1])))
            self.backward_hooks.append(MyHook(module[1], backward=True))


    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(init_weights)

    def encode(self, task_id, input_ids, token_type_ids, attention_mask, additional_features):
        if len(additional_features) == 0:
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                          attention_mask=attention_mask)
        else:
            # get embeddings

            input_embeddings = self.embed_inputs_with_additional_features(task_id=task_id, input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, additional_feature_idxs=additional_features)
            outputs = self.bert(input_ids=None, token_type_ids=token_type_ids, inputs_embeds=input_embeddings,
                                                          attention_mask=attention_mask)
            # input into bert
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

    def embed_inputs_with_additional_features(self, task_id, input_ids, position_ids, token_type_ids, additional_feature_idxs):
        # compute bert embeddings with adding embeddings for additional features before the Layernorm of the embedding layer
        # this is copied and modified from the Bert source code
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = self.bert.embeddings.inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else self.bert.embeddings.inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        additional_feature_embeds = self.additional_input_features_list[task_id](additional_feature_idxs)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + additional_feature_embeds

        embeddings = self.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.bert.embeddings.dropout(embeddings)
        return embeddings


    def forward(self, input_ids, token_type_ids, attention_mask,  premise_mask=None, hyp_mask=None, task_id=0, additional_features=[]):
        sequence_output, pooled_output = self.encode(task_id, input_ids, token_type_ids, attention_mask, additional_features=additional_features)

        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])
        if task_obj is not None:
            logits = task_obj.train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, self.dropout_list[task_id], self.scoring_list[task_id])
            return logits
        elif task_type == TaskType.Span:
            assert decoder_opt != 1
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SeqenceLabeling:

            pooled_output = sequence_output
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        elif task_type == TaskType.MaskLM:
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = sequence_output[:, :max_query, :]
                logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
            else:
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits
