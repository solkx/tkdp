import argparse
import os
import random
import torch
import numpy as np
import math
import copy
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='main',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datapath', default='data')
parser.add_argument('--dataset', default='CoNLL-2003')
parser.add_argument('--train_text', default='train_5_0_24.words')
parser.add_argument('--train_ner', default='train_5_0_24.ner')
parser.add_argument('--test_text', default='test.words')
parser.add_argument('--test_ner', default='test.ner')
parser.add_argument('--fewShot', default=5, type=int)
parser.add_argument('--base_model', default='bert')
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--max_seq_len', default=128, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--soft_kmeans', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--lr', default=3e-2,type=float)
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--use_truecase', type=str2bool, default=False)

parser.add_argument('--load_model', type=str2bool, default=False)
parser.add_argument('--load_model_name', default=None)
parser.add_argument('--load_checkpoint', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--load_dataset', type=str2bool, default=False)
parser.add_argument('--train_dataset_file', default=None)
parser.add_argument('--test_dataset_file', default=None)
parser.add_argument('--label2ids', default=None)
parser.add_argument('--id2labels', default=None)
parser.add_argument('--pre_seq_len', default=24, type=int)
parser.add_argument('--num_hidden_layers', default=12, type=int)
parser.add_argument('--num_attention_heads', default=12, type=int)
parser.add_argument('--prefix_hidden_size', default=512, type=int)
parser.add_argument('--gpu_id', default=0, type=str)
parser.add_argument('--nonPrompt', default=False, type=str2bool)
parser.add_argument('--seed', default=1005, type=int)
parser.add_argument('--is_seed', default=True, type=str2bool)
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--use_crf', default=False, type=str2bool)
parser.add_argument('--is_share', default=True, type=str2bool)
parser.add_argument('--sememe_emb', default="att", type=str, choices=["att", "knn", "random"])
parser.add_argument('--use_label_sememe', default=True, type=str2bool)
parser.add_argument('--use_text_sememe', default=True, type=str2bool)
parser.add_argument('--use_label', default=True, type=str2bool)
parser.add_argument('--use_text', default=True, type=str2bool)
parser.add_argument('--sememe_freeze', default=True, type=str2bool)
parser.add_argument('--num_training_steps_times', default=250, type=float)
parser.add_argument('--dev_batch_size', default=32, type=int)
parser.add_argument('--use_konwledge', default=True, type=str2bool)
parser.add_argument('--save_model', default=False, type=str2bool)
parser.add_argument('--save_preds', default=False, type=str2bool)
parser.add_argument('--is_train', default=True, type=str2bool)
parser.add_argument('--save_model_path', default="", type=str)
parser.add_argument('--load_model_path', default="", type=str)
parser.add_argument('--tuning_methods', default="ours", type=str, choices=["ours", "ptuning", "prompt", "prefix"])
parser.add_argument('--construct_sememe', default=True, type=str2bool)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_id}"
if args.is_seed:
    seed_torch(args.seed)
device = torch.device("cuda")
if not os.path.exists("./logger"):
    os.makedirs("./logger")
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
import time

from collections import Counter, defaultdict
from sample_few_shot import get_label_dict
from model import *
from eval_util import batch_span_eval
from data import *


def get_logger():
    pathname = f"./logger/{args.log_path}/{args.dataset}_len-{args.pre_seq_len}_num-{args.num_hidden_layers}_epoch-{args.epoch}_shot-{args.fewShot}_{args.train_text.split('_')[-2]}_seed-{args.seed}_label-{args.use_label_sememe}_text-{args.use_text_sememe}_times-{int(args.num_training_steps_times)}_{time.strftime('%m-%d_%H-%M-%S')}.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def generate_batch(batch):
    text = [F.pad(torch.tensor(x[0]), (0,max_seq_len-len(x[0])), "constant", 1) for x in batch] # batch_size * max_seq_len 
    text = pad_sequence(text, batch_first = True)
    attention_mask = [torch.cat((torch.ones_like(torch.tensor(x[0])), torch.zeros(max_seq_len-len(x[0]), dtype=torch.int64)), dim=0)
        if len(x[0]) < max_seq_len else torch.ones_like(torch.tensor(x[0]))[:max_seq_len] for x in batch]
    attention_mask = pad_sequence(attention_mask, batch_first = True)
    label = [F.pad(torch.tensor(x[1]), (0,max_seq_len-len(x[1])), "constant", -100) for x in batch]
    label = pad_sequence(label, batch_first = True)
    orig_len = [len(x[0]) for x in batch]
    none_sememes_id_list = [x[-1] for x in batch]
    init_text = [x[0] for x in batch]
    return text, attention_mask, label, orig_len, none_sememes_id_list, init_text

def train_func(processed_training_set, is_type, epoch, tokenizer, label_sentence_dicts, soft_kmeans, count_num = 0, unsup_data_iter = None):

    train_loss = 0
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    dataset_chosen = []
    data = []
    load_dic = {}
    for i,d in enumerate(processed_training_set):
        one_dataset = DataLoader(d, batch_size=BATCH_SIZE, collate_fn=generate_batch)
        data.extend(one_dataset)
        dataset_chosen.extend([i for x in range(len(one_dataset))])
    all_data_index = [i for i in range(len(dataset_chosen))]
    logger.info("shuffling sentences")
    random.shuffle(all_data_index)
    
    model.train()
    logger.info(f"total {len(all_data_index)} iters")
    for k in all_data_index:
        count_num += 1
        text, attention_mask, cls, orig_len, none_sememes_id_list, init_text = data[k]
        id2label = id2labels[dataset_chosen[k]]
        optimizer.zero_grad()
        outputs = []
        text_1, attention_mask_1, cls_1 = text.to(device), attention_mask.to(device), cls.to(device), 
        cls_2 = cls.to(device)

        loss, output, load_dic = model(text_1, init_text=init_text, is_type=is_type, epoch=epoch, load_dic=load_dic, none_sememes_id_list=none_sememes_id_list, label_id_list=label_id_list, attention_mask=attention_mask_1, labels=cls_2, dataset = dataset_chosen[k], is_train=True)
        loss.mean().backward()
        train_loss += loss.mean().item()
        outputs=output
        optimizer.step()
        preds = [[id2label[int(x)] for j,x in enumerate(y[1:orig_len[i]-1]) if int(cls[i][j + 1]) != -100] for i,y in enumerate(outputs)]
        gold = [[id2label[int(x)] for x in y[1:orig_len[i]-1] if int(x) != -100] for i,y in enumerate(cls)]
    
        bpred, bgold, bcrct, _, _, _ = batch_span_eval(preds, gold)
        total_pred += bpred
        total_gold += bgold
        total_crct += bcrct

        
        if count_num%200 == 0:
            logger.info(f"batch: {count_num}/{int(num_training_steps/N_EPOCHS)} lr: {optimizer.param_groups[0]['lr']:.9f} loss: {loss.mean().item()/BATCH_SIZE:.9f}")
            
        # Adjust the learning rate
        scheduler.step()

    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror) if (microp + micror) > 0 else 0

    return train_loss / train_num_data_point * BATCH_SIZE, microp, micror, microf1, load_dic

def test(data_, is_type, epoch, label_sentence_dicts, soft_kmeans, best_model = None, test_all_data = True, finetune = False, is_test=False):
    val_loss = 0
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    pred_list = []
    glod_list = []
    load_dic = {}
    total_pred_per_type, total_gold_per_type, total_crct_per_type = defaultdict(int), defaultdict(int), defaultdict(int)

    if test_all_data:
        dataset_chosen = []
        data = []
        for i,d in enumerate(data_):
            one_dataset = DataLoader(d, batch_size=args.dev_batch_size, collate_fn=generate_batch)
            data.extend(one_dataset)
            dataset_chosen.extend([i for x in range(len(one_dataset))])
    else:
        data, _ = generate_episode(data_, epoch,  EPISODE_NUM, TEST_SUP_CLS_NUM, test_id2label, label_sentence_dict, use_multipledata = False)
    idx = 0
    f1ss = []
    pss = []
    rss = []
    if is_test and epoch != 0:
        model_dict = model.state_dict()
        model_dict.update(best_model)
        model.load_state_dict(model_dict)
    new_model = model
    new_model.eval()
    for j, (text, attention_mask, cls, orig_len, none_sememes_id_list, init_text) in enumerate(data): 
        id2label = id2labels[dataset_chosen[j]]
        with torch.no_grad():
            text_1, attention_mask_1, cls_1 = text.to(device), attention_mask.to(device).to(device), cls.to(device)
            # we use the same dataset for training and testing
            loss, outputs, load_dic = new_model(text_1, init_text=init_text, is_type=is_type, epoch=epoch, load_dic=load_dic, none_sememes_id_list=none_sememes_id_list, label_id_list=label_id_list, attention_mask=attention_mask_1, labels=cls_1, dataset = dataset_chosen[j], is_train=False)
            val_loss += loss.mean().item()
        preds = [[id2label[int(x)] for j,x in enumerate(y[1:orig_len[i]-1]) if int(cls[i][j + 1]) != -100] for i,y in enumerate(outputs)]
        gold = [[id2label[int(x)] for x in y[1:orig_len[i]-1] if int(x) != -100] for i,y in enumerate(cls)]

        for pred in preds:
            for t,token in enumerate(pred):
                if len(token.split('I-')) == 2:
                    if t == 0:
                        pred[t] = 'O'
                        continue
                    else:
                        tag = token.split('I-')[1]
                        if len(pred[t-1]) == 1:
                            pred[t] = 'O'
                        else:
                            if tag != pred[t-1].split('-')[1]:
                                pred[t] = 'O'  

        pred_list.append(preds)
        glod_list.append(gold)
        bpred, bgold, bcrct, pred_span_per_type, gold_span_per_type, crct_span_per_type = batch_span_eval(preds, gold)
        total_pred += bpred
        total_gold += bgold
        total_crct += bcrct
        for x in pred_span_per_type:
            total_pred_per_type[x] += pred_span_per_type[x]
            total_gold_per_type[x] += gold_span_per_type[x]
            total_crct_per_type[x] += crct_span_per_type[x]

    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror) if (microp + micror) > 0 else 0
    microp_per_type, micror_per_type, microf1_per_type = {}, {}, {}
    for x in total_pred_per_type:
        microp_per_type[x] = total_crct_per_type[x]/total_pred_per_type[x] if total_pred_per_type[x] > 0 else 0
        micror_per_type[x] = total_crct_per_type[x]/total_gold_per_type[x] if total_gold_per_type[x] > 0 else 0
        microf1_per_type[x] = 2*microp_per_type[x]*micror_per_type[x]/(microp_per_type[x]+micror_per_type[x]) if (microp_per_type[x]+micror_per_type[x]) > 0 else 0
    if is_test and args.save_preds:
        with open(f"./logger/{args.log_path}/{args.dataset}_len-{args.pre_seq_len}_num-{args.num_hidden_layers}_epoch-{args.epoch}_shot-{args.fewShot}_{args.train_text.split('_')[-1]}_seed-{args.seed}_label-{args.use_label_sememe}_text-{args.use_text_sememe}_times-{int(args.num_training_steps_times)}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(pred_list))
        with open(f"logger/{args.log_path}/glod.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(glod_list))   
    return val_loss / test_num_data_point * 64, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type, load_dic

if __name__ == "__main__":
    logger = get_logger()
    logger.info(args)

    train_text_file = args.train_text
    train_ner_file = args.train_ner
    if '.' not in args.train_text:
        args.train_text = train_text_file + f'_{round}.words'
        args.train_ner = train_ner_file + f'_{round}.ner'
    logger.info(f"train text is {args.train_text}")
    datasets = args.dataset.split('_')
    logger.info(datasets)
    train_texts = [os.path.join(args.datapath, dataset, args.train_text) for dataset in datasets]
    train_ners = [os.path.join(args.datapath, dataset, args.train_ner) for dataset in datasets]
    dev_texts = [os.path.join(args.datapath, dataset, f"dev_{args.fewShot}.words") for dataset in datasets]
    dev_ners = [os.path.join(args.datapath, dataset, f"dev_{args.fewShot}.ner") for dataset in datasets]
    test_texts = [os.path.join(args.datapath, dataset, args.test_text) for dataset in datasets]
    test_ners = [os.path.join(args.datapath, dataset, args.test_ner) for dataset in datasets]

    max_seq_len = args.max_seq_len
    base_model = args.base_model

    if base_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif base_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    label2ids, id2labels = [], []
    processed_training_set, train_label_sentence_dicts, processed_dev_set, dev_label_sentence_dicts, processed_test_set, test_label_sentence_dicts = [], [], [], [], [], []
    if not args.construct_sememe:
        with open(f"./data/{args.dataset}/label_sememes_id.json", "r", encoding="utf-8") as f:
            label_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_id_{args.train_text.split('.')[0]}.json", "r", encoding="utf-8") as f:
            train_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_id_dev_{args.fewShot}.json", "r", encoding="utf-8") as f:
            dev_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_id_test.json", "r", encoding="utf-8") as f:
            test_id_list = json.loads(f.read())
    else:
        with open(f"./data/{args.dataset}/label_sememes_tree.json", "r", encoding="utf-8") as f:
            label_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_tree_{args.train_text.split('.')[0]}.json", "r", encoding="utf-8") as f:
            train_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_tree_dev_{args.fewShot}.json", "r", encoding="utf-8") as f:
            dev_id_list = json.loads(f.read())
        with open(f"./data/{args.dataset}/text_sememes_tree_test.json", "r", encoding="utf-8") as f:
            test_id_list = json.loads(f.read())
    if not args.load_dataset:
        for train_text, train_ner, dev_text, dev_ner, test_text, test_ner in zip(train_texts, train_ners, dev_texts, dev_ners, test_texts, test_ners):
            with open(train_ner) as fner, open(train_text) as f:
                train_ner_tags, train_words = fner.readlines(), f.readlines() 
            with open(dev_ner) as fner, open(dev_text) as f:
                dev_ner_tags, dev_words = fner.readlines(), f.readlines() 
            with open(test_ner) as fner, open(test_text) as f:
                test_ner_tags, test_words = fner.readlines(), f.readlines()      
            label2id, id2label = get_label_dict([train_ner_tags, dev_ner_tags, test_ner_tags])
            
            label2ids.append(label2id)
            id2labels.append(id2label)

            train_ner_tags, train_words, train_label_sentence_dict = process_data(train_ner_tags, train_words, tokenizer, label2id, max_seq_len,base_model=base_model,use_truecase=args.use_truecase)
            dev_ner_tags, dev_words, dev_label_sentence_dict = process_data(dev_ner_tags, dev_words, tokenizer, label2id, max_seq_len,base_model=base_model,use_truecase=args.use_truecase)
            test_ner_tags, test_words, test_label_sentence_dict = process_data(test_ner_tags, test_words, tokenizer, label2id, max_seq_len,base_model=base_model,use_truecase=args.use_truecase)

            sub_train_ = [[train_words[i], train_ner_tags[i], train_id_list[i]] for i in range(len(train_ner_tags))]
            sub_dev_ = [[dev_words[i], dev_ner_tags[i], dev_id_list[i]] for i in range(len(dev_ner_tags))] 
            sub_valid_ = [[test_words[i], test_ner_tags[i], test_id_list[i]] for i in range(len(test_ner_tags))] 

            train_label_sentence_dicts.append(train_label_sentence_dict)
            dev_label_sentence_dicts.append(dev_label_sentence_dict)
            test_label_sentence_dicts.append(test_label_sentence_dict)

            processed_training_set.append(sub_train_) 
            processed_dev_set.append(sub_dev_)
            processed_test_set.append(sub_valid_) 

    if args.load_dataset:
        start_time = time.time()
        logger.info("start loading training dataset!")
        processed_training_set = np.load(args.train_dataset_file,allow_pickle=True)
        logger.info("start loading dev dataset!")
        processed_dev_set = np.load(args.dev_dataset_file,allow_pickle=True)
        logger.info("start loading test dataset!")
        processed_test_set = np.load(args.test_dataset_file,allow_pickle=True)
        logger.info("start loading label ids!")
        label2ids = np.load(args.label2ids,allow_pickle=True)
        id2labels = np.load(args.id2labels,allow_pickle=True)
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        logger.info(f"finish loading datasets! | time in {mins} minutes, {secs} seconds")
        train_label_sentence_dict = None
        test_label_sentence_dict = None
        

    dataset_label_nums = [len(x) for x in label2ids]
    logger.info(f"dataset label nums: {dataset_label_nums}")
    train_num_data_point = sum([len(sub_train_) for sub_train_ in processed_training_set])
    logger.info(f"number of all training data points: {train_num_data_point}")
    dev_num_data_point = sum([len(sub_train_) for sub_train_ in processed_dev_set])
    logger.info(f"number of all deving data points: {dev_num_data_point}")
    test_num_data_point = sum([len(sub_train_) for sub_train_ in processed_test_set])
    logger.info(f"number of all testing data points: {test_num_data_point}")



    LOAD_MODEL = args.load_model
    if LOAD_MODEL:
        print("="*100)
        label_sememe_dict = torch.load("./pretrained_models/label_sememe.dict")
        roberta = torch.load("./pretrained_models/roberta.dict")

        if base_model == 'roberta':
            model = RobertaNER.from_pretrained('roberta-base', args=args, dataset_label_nums = dataset_label_nums, output_attentions=False, output_hidden_states=False, multi_gpus=True)
        elif base_model == 'bert':
            model = BertNER.from_pretrained('bert-base-uncased', args=args, dataset_label_nums = dataset_label_nums, output_attentions=False, output_hidden_states=False, multi_gpus=True)
        model_dict = model.state_dict()
        pretrain_dic = {f"label_sememe_fusion.label_sememe.{k}":v for k,v in label_sememe_dict.items() if f"label_sememe_fusion.label_sememe.{k}" in model_dict} 
        pretrain_dic.update({f"roberta.{k}":v for k,v in roberta.items() if f"roberta.{k}" in model_dict})
        model_dict.update(pretrain_dic)
        model.load_state_dict(model_dict)
        model.dataset_label_nums = dataset_label_nums
        model.classifiers = torch.nn.ModuleList([torch.nn.Linear(model.config.hidden_size, x) for x in dataset_label_nums])
    else:
        if base_model == 'roberta':
            model = RobertaNER.from_pretrained('roberta-base', args=args, dataset_label_nums = dataset_label_nums, output_attentions=False, output_hidden_states=False, multi_gpus=True)
        elif base_model == 'bert':
            model = BertNER.from_pretrained('bert-base-uncased', args=args, dataset_label_nums = dataset_label_nums, output_attentions=False, output_hidden_states=False, multi_gpus=True)
        model.dataset_label_nums = dataset_label_nums
        model.classifiers = torch.nn.ModuleList([torch.nn.Linear(model.config.hidden_size, x) for x in dataset_label_nums])
    logger.info(f"let's use {torch.cuda.device_count()} GPUs!")
    model.to(device)
    # print(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    num_training_steps = math.ceil(args.num_training_steps_times * N_EPOCHS * int(sum([len(sub_train_) for sub_train_ in processed_training_set]) / BATCH_SIZE))
    num_warmup_steps = int(args.warmup_proportion * num_training_steps)
    logger.info(f"num training steps: {num_training_steps}")
    logger.info(f"num warmup steps: {num_warmup_steps}")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    SOFT_KMEANS = args.soft_kmeans

    start_epoch = 0
    LOAD_CHECKPOINT = args.load_checkpoint
    if LOAD_CHECKPOINT:
        logger.info("start loading checkpoint")
        start_time = time.time()
        state = torch.load(args.load_model_name)


        model.load_state_dict(state['state_dict'])
        start_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        logger.info(f"loaded from checkpoint - learning rate: {optimizer.param_groups[0]['lr']:.9f} | time in {mins} minutes, {secs} seconds")
        start_count_num = state['count_num']

    best_dev_f1 = 0.00
    bestModel = None
    bestEpoch = 0
    if args.is_train:
        for epoch in range(start_epoch, N_EPOCHS):

            start_time = time.time()
            if args.load_checkpoint and epoch == start_epoch:
                train_loss, microp, micror, microf1, train_load_dic = train_func(processed_training_set, "train", epoch, tokenizer, train_label_sentence_dicts, soft_kmeans = SOFT_KMEANS, count_num = start_count_num)
            else:
                train_loss, microp, micror, microf1, train_load_dic = train_func(processed_training_set, "train", epoch, tokenizer, train_label_sentence_dicts, soft_kmeans = SOFT_KMEANS)
            secs = time.time() - start_time
            logger.info(f'Loss: {train_loss:.4f}(train)\t{secs} seconds')
            dev_loss, microp, micror, dev_microf1, microp_per_type, micror_per_type, microf1_per_type, dev_load_dic = test(processed_dev_set, "dev", epoch, dev_label_sentence_dicts, soft_kmeans = SOFT_KMEANS)
            logger.info(f'F1: {dev_microf1 * 100:.1f}%(val)')
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            if best_dev_f1 < dev_microf1 or epoch == 0:
                best_dev_f1 = dev_microf1
                start_time = time.time()
                if epoch == 0 and not args.construct_sememe:
                    valid_loss, microp, micror, test_microf1, microp_per_type, micror_per_type, microf1_per_type, test_load_dic = test(processed_test_set, "test", epoch, test_label_sentence_dicts, best_model = model, soft_kmeans = SOFT_KMEANS, is_test=True)
                    secs = time.time() - start_time
                    logger.info(f'F1: {test_microf1 * 100:.1f}%(test)\t{secs} seconds')
                bestModel = copy.deepcopy(model.state_dict())
                bestEpoch = epoch
            if epoch == 0 and not args.construct_sememe:
                torch.save(train_load_dic, f"./data/{args.dataset}/train_{args.fewShot}.emb")
                torch.save(dev_load_dic, f"./data/{args.dataset}/dev_{args.fewShot}.emb")
                torch.save(test_load_dic, f"./data/{args.dataset}/test_{args.fewShot}.emb")


            logger.info(f"Epoch: {epoch + 1} | time in {mins} minutes, {secs} seconds")
    
        if args.save_model:
            start_time = time.time()
            valid_loss, microp, micror, test_microf1, microp_per_type, micror_per_type, microf1_per_type, test_load_dic = test(processed_test_set, "test", bestEpoch, test_label_sentence_dicts, best_model = bestModel, soft_kmeans = SOFT_KMEANS, is_test=True) 
            secs = time.time() - start_time
            logger.info(f'F1: {test_microf1 * 100:.1f}%(test)\t{secs} seconds')
            torch.save(bestModel, f"./{args.save_model_path}")
        else:
            start_time = time.time()
            valid_loss, microp, micror, test_microf1, microp_per_type, micror_per_type, microf1_per_type, test_load_dic = test(processed_test_set, "test", bestEpoch, test_label_sentence_dicts, best_model = bestModel, soft_kmeans = SOFT_KMEANS, is_test=True)
            secs = time.time() - start_time
            logger.info(f'F1: {test_microf1 * 100:.1f}%(test)\t{secs} seconds')
    else:
        best_model = torch.load(f"./{args.load_model_path}")
        model_dict = model.state_dict()
        model_dict.update(best_model)
        model.load_state_dict(model_dict)
        start_time = time.time()
        valid_loss, microp, micror, test_microf1, microp_per_type, micror_per_type, microf1_per_type, test_load_dic = test(processed_test_set, "test", bestEpoch, test_label_sentence_dicts, best_model = best_model, soft_kmeans = SOFT_KMEANS, is_test=True)
        secs = time.time() - start_time
        logger.info(f'F1: {test_microf1 * 100:.1f}%(test)\t{secs} seconds')

    logger.info(f"f1 scores: {[test_microf1]}")      