# naive token-level classification model (including bert-based and roberta-based)

import torch
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import List, Optional
import copy
import OpenHowNet
import json
from torchcrf import CRF

class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, hidden_size):
        super().__init__()
        if config.tuning_methods != "ours":
            self.embedding = torch.nn.Embedding(config.pre_seq_len, hidden_size)
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * hidden_size)
    def forward(self, prefix: torch.Tensor):
        past_key_values = self.embedding(prefix)
        return past_key_values

class LabelSememeFusionPrompt(nn.Module):
    def __init__(self, config, hidden_size, args, dataset_label_nums, roberta_embedding):
        super(LabelSememeFusionPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.dataset_label_nums = dataset_label_nums
        self.dropout = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.use_label_sememe = args.use_label_sememe
        self.use_text_sememe = args.use_text_sememe
        self.is_share = args.is_share
        self.use_label = args.use_label
        self.use_text = args.use_text
        self.sememe_emb = args.sememe_emb
        self.label_sememe = SememeEmbeddingKNN(config, hidden_size, self.use_label_sememe, self.sememe_emb, roberta_embedding, self.is_share)
        self.text_sememe = SememeEmbeddingKNN(config, hidden_size, self.use_text_sememe, self.sememe_emb, roberta_embedding, self.is_share)
        self.lstm_head = torch.nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.0,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size))
        self.args = args

    def forward(self, prompt, label_id_list, text_id_list, init_label_feature=None, init_text_feature=None):
        """
            token_feature: [batch_size, context_seq_len, head_num, hidden_dim]
            label_feature: [batch_size, class_num, hidden_dim]
            text_dewature: [batch_size, class_num, hidden_dim]
        """
        if init_label_feature is None:
            init_label_feature = self.label_sememe(label_id_list) # class_num, 768
        if init_text_feature is None:
            init_text_feature = self.text_sememe(text_id_list) # bs, 768
        batch_size, seq_len, head_num, _ = prompt.shape
        label_num = init_label_feature.shape[0]
        prompt_fc = self.fc_1(prompt) # [bs, len, head_num, hidden]
        if self.use_label: 
            label_feature = self.fc_2(init_label_feature) # [label_num, hidden]
            prompt_fc_label = prompt_fc[:, :prompt_fc.shape[1]//2, :, :]
            if not self.args.nonPrompt:
                scores_label = torch.einsum("abcd,ed->abce", prompt_fc_label, label_feature)  # [bs, context_seq_len, head_num, label_num]
                scores_label = torch.softmax(scores_label, dim=-1)
                weighted_label_feature = label_feature.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len // 2, head_num, 1, 1) * scores_label.unsqueeze(-1) # [bs, len, head_num, label_num, hidden_dim]
                # [bs, context_seq_len, class_num, hidden_dim]
                weighted_label_feature_sum = torch.sum(weighted_label_feature, dim=-2)
                # [bs, context_seq_len, class_num, hidden_dim]
                weighted_label_feature_sum = prompt_fc_label + weighted_label_feature_sum
            else:
                weighted_label_feature = label_feature.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len // 2, head_num, 1, 1)
                weighted_label_feature_sum = torch.sum(weighted_label_feature, dim=-2)
            fused_label_feature = torch.tanh(self.fc_5(weighted_label_feature_sum))
        else:
            fused_label_feature = prompt[:, :prompt.shape[1]//2, :, :]

        # [bs, context_seq_len, class_num, hidden_dim]
        if self.use_text:
            text_feature = self.fc_3(init_text_feature).unsqueeze(1) # [batch_size, 1, hidden]
            prompt_fc_text = prompt_fc[:, prompt_fc.shape[1]//2:, :, :]
            if not self.args.nonPrompt:
                scores_text = torch.einsum("abcd,aed->abce", prompt_fc_text, text_feature)  # [bs, context_seq_len, head_num, 1]
                scores_text = torch.softmax(scores_text, dim=-1)
                weighted_text_feature = text_feature.unsqueeze(1).unsqueeze(1).repeat(1, seq_len // 2, head_num, 1, 1) * scores_text.unsqueeze(-1) # [bs, len, head_num, 1, hidden_dim]
                # [bs, context_seq_len, class_num, hidden_dim]
                weighted_text_feature_sum = torch.sum(weighted_text_feature, dim=-2)
                weighted_text_feature_sum = prompt_fc_text + weighted_text_feature_sum
            else:
                weighted_text_feature = text_feature.unsqueeze(1).unsqueeze(1).repeat(1, seq_len // 2, head_num, 1, 1)
                weighted_text_feature_sum = torch.sum(weighted_text_feature, dim=-2)
            fused_text_feature = torch.tanh(self.fc_4(weighted_text_feature_sum))
        else:
            fused_text_feature = prompt[:, prompt.shape[1]//2:, :, :]
        fused_feature = torch.cat([fused_label_feature, fused_text_feature], dim=1)
        if self.args.tuning_methods == "ptuning":
            fused_feature = self.mlp_head(self.lstm_head(fused_feature.squeeze(2))[0])
        return fused_feature, init_text_feature, init_label_feature

class SememeEmbeddingKNN(nn.Module):
    def __init__(self, config, hidden_size, use_sememe, sememe_emb, roberta_embedding, is_share):
        super(SememeEmbeddingKNN, self).__init__()
        self.is_share = is_share
        if not self.is_share:
            self.embedding = torch.nn.Embedding(config.vocab_size, hidden_size)
        else:
            self.embedding = roberta_embedding
        self.use_sememe = use_sememe
        self.sememe_emb = sememe_emb
        self.hidden_size = hidden_size

    def forward(self, label_id):
        sememe_s = []
        for label_words in label_id:
            span = []
            for word_list in label_words:
                if not word_list:
                    continue
                if not self.is_share:
                    s = self.embedding(torch.tensor(word_list[0]).cuda())
                else:
                    # print(word_list[0])
                    s = self.embedding(torch.tensor(word_list[0]).cuda().unsqueeze(0)).squeeze(0)
                senses_id_list = word_list[1]
                if self.use_sememe:
                    if len(senses_id_list) != 0:
                        if not self.is_share:
                            sememe_tensor = torch.cat([self.embedding(torch.tensor(sm).cuda()) for sm in senses_id_list], dim=0)
                        else:
                            sememe_tensor = torch.cat([self.embedding(torch.tensor(sm).cuda().unsqueeze(0)).squeeze(0) for sm in senses_id_list], dim=0)
                    if self.sememe_emb == "att":
                        distance = F.pairwise_distance(s, sememe_tensor, p=2)
                        attentionSocre = torch.softmax(distance, dim=0)
                        attentionSememeTensor = torch.einsum("a,ab->ab", attentionSocre, sememe_tensor)
                        span.append(torch.cat([attentionSememeTensor.mean(0).unsqueeze(0), s], dim=0).mean(0))
                    elif self.sememe_emb == "knn":
                        distance = F.pairwise_distance(s, sememe_tensor, p=2)
                        span.append(torch.cat([torch.stack([sememe_tensor[idx] for idx in torch.sort(distance, descending=True)[1][:3]], dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
                    else:
                        span.append(torch.cat([torch.stack([sememe_tensor[random.randint(0, sememe_tensor.shape[0])] for idx in range(3)], dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
                else:
                    span.append(s.mean(dim=0))
            if len(span) != 0:
                sememe_s.append(torch.stack(span, dim=0).mean(0))
            else:
                sememe_s.append(torch.ones(self.hidden_size).cuda())
        return torch.stack(sememe_s, dim=0)


        


class RobertaNER(RobertaForTokenClassification):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, args, dataset_label_nums, multi_gpus=False):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dataset_label_nums = dataset_label_nums
        self.multi_gpus = multi_gpus
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
        self.background = torch.nn.Parameter(torch.zeros(1)- 2., requires_grad=True)

        self.init_weights() 
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.pre_seq_len = args.pre_seq_len
        self.n_layer = args.num_hidden_layers
        self.n_head = args.num_attention_heads
        self.n_embd = config.hidden_size // args.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(args, config.hidden_size)
        self.max_seq_len = args.max_seq_len
        self.sememe_fusion = LabelSememeFusionPrompt(config, config.hidden_size, args, self.dataset_label_nums[0], self.roberta.embeddings.word_embeddings)
        if args.sememe_freeze:
            for param in self.sememe_fusion.label_sememe.parameters():
                param.requires_grad = False
            for param in self.sememe_fusion.text_sememe.parameters():
                param.requires_grad = False
        self.use_konwledge = args.use_konwledge
        self.dataset = args.dataset
        self.few_shot_num = args.fewShot
        self.args = args
        self.roberta_num_hidden_layers = config.num_hidden_layers
        if self.args.use_crf:
            self.crf = CRF(self.dataset_label_nums[0], batch_first=True)

    def forward(self, input_ids,
        init_text,
        is_type,
        epoch,
        load_dic,
        none_sememes_id_list,
        label_id_list,
        text_id_list=None,
        attention_mask=None,
        dataset = 0,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_logits=False,
        is_train=True):
        batch_size = input_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        if self.args.tuning_methods == "ours":
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                -1
            ) # [4, 48, 48, 12, 64]
        else:
            past_key_values = past_key_values.unsqueeze(2)
        if self.use_konwledge:
            if epoch == 0:
                past_key_values, text_feature, label_feature = self.sememe_fusion(past_key_values, label_id_list, none_sememes_id_list)
                for iii in range(batch_size):
                    load_dic[str(init_text[iii])] = text_feature[iii]
                torch.save(label_feature, f"./data/{self.dataset}/label_{self.few_shot_num}.emb")
            else:
                if is_type == "train":
                    load_d = torch.load(f"./data/{self.dataset}/train_{self.few_shot_num}.emb")
                if is_type == "dev":
                    load_d = torch.load(f"./data/{self.dataset}/dev_{self.few_shot_num}.emb")
                if is_type == "test":
                    load_d = torch.load(f"./data/{self.dataset}/test_{self.few_shot_num}.emb")
                label_feature = torch.load(f"./data/{self.dataset}/label_{self.few_shot_num}.emb").cuda()
                text_feature = torch.stack([load_d[str(init_text[iii])] for iii in range(batch_size)], dim=0).cuda()
                past_key_values = self.sememe_fusion(past_key_values, label_id_list, none_sememes_id_list, label_feature, text_feature)[0]
        if self.args.tuning_methods == "ours" or self.args.tuning_methods == "prefix":
            if self.args.tuning_methods == "prefix":
                past_key_values = past_key_values.expand(-1,-1,self.n_layer * 2,-1)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            ) # [4, 48, 48, 12, 64]
            past_key_values = torch.cat((past_key_values, torch.zeros(batch_size, self.pre_seq_len, (self.roberta_num_hidden_layers-self.n_layer)*2, self.n_head, self.n_embd).cuda()), dim=2)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                past_key_values=past_key_values
            )
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
        else:
            if self.args.tuning_methods != "ptunig":
                past_key_values = past_key_values.squeeze(2)
            raw_embedding = self.roberta.embeddings(
                input_ids=input_ids,
                position_ids=position_ids
            )
            inputs_embeds = torch.cat((past_key_values, raw_embedding), dim=1)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).cuda()
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.roberta(
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
            sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        attention_mask = attention_mask[:,self.pre_seq_len:].contiguous()
        if not self.args.use_crf:
            outputs = torch.argmax(logits, dim=2)
        else:
            outputs = self.crf.decode(logits)
        loss = torch.tensor([0.0])
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.dataset_label_nums[dataset])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                if not self.args.use_crf:
                    loss = loss_fct(active_logits, active_labels)
                else:
                    if is_train:
                        active_labels_numpy = active_labels.cpu().numpy()
                        active_labels_numpy[(active_labels_numpy == 0)] = -1
                        active_labels_numpy[(active_labels_numpy == -100)] = 0
                        labels_mask = torch.from_numpy(active_labels_numpy).bool().cuda()
                        active_labels_numpy[(active_labels_numpy == -1)] = 0
                        new_l = torch.from_numpy(active_labels_numpy).long().cuda()
                        loss = -self.crf(active_logits.unsqueeze(0), new_l.unsqueeze(0), mask=labels_mask.unsqueeze(0), reduction='mean')
            else:
                if self.args.use_crf:
                    if is_train:
                        active_labels_numpy = active_labels.cpu().numpy()
                        active_labels_numpy[(active_labels_numpy == 0)] = -1
                        active_labels_numpy[(active_labels_numpy == -100)] = 0
                        labels_mask = torch.from_numpy(active_labels_numpy).bool().cuda()
                        active_labels_numpy[(active_labels_numpy == -1)] = 0
                        new_l = torch.from_numpy(active_labels_numpy).long().cuda()
                        loss = -self.crf(logits, new_l, mask=labels_mask, reduction='mean')
                else:
                    loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
            if output_logits:
                return loss, outputs, logits, load_dic
            else:
                return loss, outputs, load_dic
        else:
            return outputs, load_dic



class BertNER(BertForTokenClassification):
    config_class = BertConfig

    def __init__(self, config, args, dataset_label_nums, multi_gpus=False):
        super().__init__(config)
        self.bert = BertModel(config)

        self.dataset_label_nums = dataset_label_nums
        self.multi_gpus = multi_gpus
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
        self.background = torch.nn.Parameter(torch.zeros(1)- 2., requires_grad=True)

        self.init_weights() 
        for param in self.bert.parameters():
            param.requires_grad = False
        self.pre_seq_len = args.pre_seq_len
        self.n_layer = args.num_hidden_layers
        self.n_head = args.num_attention_heads
        self.n_embd = config.hidden_size // args.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(args, config.hidden_size)
        self.max_seq_len = args.max_seq_len
        self.sememe_fusion = LabelSememeFusionPrompt(config, config.hidden_size, args, self.dataset_label_nums[0], self.bert.embeddings.word_embeddings)
        if args.sememe_freeze:
            for param in self.sememe_fusion.label_sememe.parameters():
                param.requires_grad = False
            for param in self.sememe_fusion.text_sememe.parameters():
                param.requires_grad = False
        self.use_konwledge = args.use_konwledge
        self.dataset = args.dataset
        self.few_shot_num = args.fewShot
        self.args = args
        self.bert_num_hidden_layers = config.num_hidden_layers
        if self.args.use_crf:
            self.crf = CRF(self.dataset_label_nums[0], batch_first=True)

    def forward(self, input_ids,
        init_text,
        is_type,
        epoch,
        load_dic,
        none_sememes_id_list,
        label_id_list,
        text_id_list=None,
        attention_mask=None,
        dataset = 0,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_logits=False,
        is_train=True):
        batch_size = input_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        if self.args.tuning_methods == "ours":
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                -1
            ) # [4, 48, 48, 12, 64]
        else:
            past_key_values = past_key_values.unsqueeze(2)
        if self.use_konwledge:
            if epoch == 0:
                past_key_values, text_feature, label_feature = self.sememe_fusion(past_key_values, label_id_list, none_sememes_id_list)
                for iii in range(batch_size):
                    load_dic[str(init_text[iii])] = text_feature[iii]
                torch.save(label_feature, f"./data/{self.dataset}/label_{self.few_shot_num}.emb")
            else:
                if is_type == "train":
                    load_d = torch.load(f"./data/{self.dataset}/train_{self.few_shot_num}.emb")
                if is_type == "dev":
                    load_d = torch.load(f"./data/{self.dataset}/dev_{self.few_shot_num}.emb")
                if is_type == "test":
                    load_d = torch.load(f"./data/{self.dataset}/test_{self.few_shot_num}.emb")
                label_feature = torch.load(f"./data/{self.dataset}/label_{self.few_shot_num}.emb").cuda()
                text_feature = torch.stack([load_d[str(init_text[iii])] for iii in range(batch_size)], dim=0).cuda()
                past_key_values = self.sememe_fusion(past_key_values, label_id_list, none_sememes_id_list, label_feature, text_feature)[0]
        if self.args.tuning_methods == "ours" or self.args.tuning_methods == "prefix":
            if self.args.tuning_methods == "prefix":
                past_key_values = past_key_values.expand(-1,-1,self.n_layer * 2,-1)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            ) # [4, 48, 48, 12, 64]
            past_key_values = torch.cat((past_key_values, torch.zeros(batch_size, self.pre_seq_len, (self.bert_num_hidden_layers-self.n_layer)*2, self.n_head, self.n_embd).cuda()), dim=2)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                past_key_values=past_key_values
            )
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
        else:
            if self.args.tuning_methods != "ptunig":
                past_key_values = past_key_values.squeeze(2)
            raw_embedding = self.bert.embeddings(
                input_ids=input_ids,
                position_ids=position_ids
            )
            inputs_embeds = torch.cat((past_key_values, raw_embedding), dim=1)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).cuda()
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.bert(
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
            sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        attention_mask = attention_mask[:,self.pre_seq_len:].contiguous()
        if not self.args.use_crf:
            outputs = torch.argmax(logits, dim=2)
        else:
            outputs = self.crf.decode(logits)
        loss = torch.tensor([0.0])
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.dataset_label_nums[dataset])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                if not self.args.use_crf:
                    loss = loss_fct(active_logits, active_labels)
                else:
                    if is_train:
                        active_labels_numpy = active_labels.cpu().numpy()
                        active_labels_numpy[(active_labels_numpy == 0)] = -1
                        active_labels_numpy[(active_labels_numpy == -100)] = 0
                        labels_mask = torch.from_numpy(active_labels_numpy).bool().cuda()
                        active_labels_numpy[(active_labels_numpy == -1)] = 0
                        new_l = torch.from_numpy(active_labels_numpy).long().cuda()
                        loss = -self.crf(active_logits.unsqueeze(0), new_l.unsqueeze(0), mask=labels_mask.unsqueeze(0), reduction='mean')
            else:
                if self.args.use_crf:
                    if is_train:
                        active_labels_numpy = active_labels.cpu().numpy()
                        active_labels_numpy[(active_labels_numpy == 0)] = -1
                        active_labels_numpy[(active_labels_numpy == -100)] = 0
                        labels_mask = torch.from_numpy(active_labels_numpy).bool().cuda()
                        active_labels_numpy[(active_labels_numpy == -1)] = 0
                        new_l = torch.from_numpy(active_labels_numpy).long().cuda()
                        loss = -self.crf(logits, new_l, mask=labels_mask, reduction='mean')
                else:
                    loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
            if output_logits:
                return loss, outputs, logits, load_dic
            else:
                return loss, outputs, load_dic
        else:
            return outputs, load_dic