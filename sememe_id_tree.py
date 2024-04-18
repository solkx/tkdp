import json
import OpenHowNet
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
from tqdm import tqdm
import os
# OpenHowNet.download()

def layer_per_pro(root, ids2nodes, nodes2ids, edges, root_word=""):
    if not root_word:
        root_word = f'{str(root["name"])}_{len(ids2nodes)}'
    if "children" in root:
        children = root["children"]
        for child in children:
            child_word = f'{str(child["name"])}_{len(ids2nodes)}'
            ids2nodes[len(ids2nodes)] = child_word
            nodes2ids[child_word] = len(nodes2ids)
            edges[f"{nodes2ids[root_word]}-{nodes2ids[child_word]}"] = child["role"]
            ids2nodes, nodes2ids, edges = layer_per_pro(child, ids2nodes, nodes2ids, edges, child_word)
    return ids2nodes, nodes2ids, edges


    

def tree2adj(tree_list):
    sense_list = []
    for pre_sense_tree in tree_list:
        root = pre_sense_tree["sememes"]
        real_root_word = f'{"|".join(str(root["name"]).split("|")[1:])}_0'
        sense_ids2nodes = {0:real_root_word}
        sense_nodes2ids = {real_root_word:0}
        sense_edges = {}
        # print(pre_sense_tree)
        ids2nodes, nodes2ids, edges = layer_per_pro(root, sense_ids2nodes, sense_nodes2ids, sense_edges, real_root_word)
        # print(ids2nodes)
        # print(nodes2ids)
        # print(edges)
        assert len(ids2nodes) == len(nodes2ids)
        init_adj = torch.zeros((len(ids2nodes), len(ids2nodes)))
        for edge in edges:
            i, j = edge.split("-")
            init_adj[int(i), int(j)] = 1
        nodes_list = [node for node in nodes2ids]
        assert len(nodes_list) == init_adj.shape[0]
        sense_list.append([nodes_list, init_adj.tolist()])
    # exit()
    return sense_list
            

def main(dataset):
    
    with open("./data/nature_form_label.json", "r", encoding="utf-8") as f:
        data_set_all_label = json.loads(f.read())
    all_label =  data_set_all_label[dataset]["all_type"]
    print(all_label)

    hownet_dict = OpenHowNet.HowNetDict()

    label_list = []
    label_id_list = []
    for label in all_label:
        if "B" in label:
            label_word = f"{startToken} begin of {' '.join(label.split('-')[-1].split('_')).lower()} tag".split(" ")
        elif "I" in label:
            label_word = f"{startToken} inside of {' '.join(label.split('-')[-1].split('_')).lower()} tag".split(" ")
        else:
            label_word = f"{startToken} others".split(" ")
        words_list = []
        words_id_list = []
        label_token = tokenizer.tokenize(" ".join(label_word))
        label_word = label_word
        num = 0
        print(label_token)
        align = []
        curr_wrd = 1 # start at 1, b/c of CLS
        buf = []
        buf_o = []
        for i in range(1, len(label_token)): # ignore [SEP] final token
            if label_token[i].startswith("##"):
                strpd = label_token[i][2:]
                strpd_o = label_token[i] 
            else:
                strpd = label_token[i]
                strpd_o = label_token[i]
            buf.append(strpd)
            buf_o.append(strpd_o)
            fwrd = ''.join(buf)
            wrd = label_word[curr_wrd].lower()
            if fwrd == wrd or fwrd == "[UNK]":
                align.append([wrd, buf_o])
                curr_wrd += 1
                buf = []
                buf_o = []
        for item in align:
            word = item[0]
            p_word_list = item[1]
            for p_word in p_word_list:
                item_list = []
                item_id_list = []
                word = word.lower()
                pretree_list = hownet_dict.get_sememes_by_word(word=word, display='dict')
                sememe_adj_list = tree2adj(pretree_list)
                item_list.append([p_word])
                item_id_list.append(tokenizer.convert_tokens_to_ids([p_word]))
                senses = []
                senses_id = []
                for sememe_adj in sememe_adj_list:
                    sememe = sememe_adj[0]
                    adj = sememe_adj[-1]
                    sememes = []
                    sememes_id = []
                    for item in sememe:
                        item = str(item)
                        if not tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]):
                            print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]))
                        sememes_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]))
                        sememes.append(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:])
                    senses.append(sememes)
                    senses_id.append([sememes_id, adj])
                item_list.append(senses)
                item_id_list.append(senses_id)
            words_list.append(item_list)
            words_id_list.append(item_id_list)
        label_list.append(words_list)
        label_id_list.append(words_id_list)
    with open(f"./data/{dataset}/label_sememes_tree.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(label_id_list))


    path = []
    for fileName in os.listdir(f"./data/{dataset}"):
        if ".ner" not in fileName:
            continue
        if fileName.split(".")[0] not in path:
            path.append(fileName.split(".")[0])
    for name in path:
        print(name)
        with open(f"./data/{dataset}/{name}.words", "r", encoding="utf-8") as f:
            sentences_list = f.read().split("\n")

        sentence_list = []
        sentence_id_list = []
        line = 0
        for sentence in tqdm(sentences_list):
            new_sentence = []
            line += 1
            w_list = f"{startToken} {sentence}".split(" ")
            words_list = []
            words_id_list = []
            w_token = tokenizer.tokenize(" ".join(w_list))
            w_list = w_list
            num = 0
            align = []
            curr_wrd = 1 # start at 1, b/c of CLS
            buf = []
            buf_o = []
            for i in range(1, len(w_token)): # ignore [SEP] final token
                if w_token[i].startswith("##"):
                    strpd = w_token[i][2:]
                    strpd_o = w_token[i] 
                else:
                    strpd = w_token[i]
                    strpd_o = w_token[i]
                buf.append(strpd)
                buf_o.append(strpd_o)
                fwrd = ''.join(buf)
                wrd = w_list[curr_wrd].lower()
                if fwrd == wrd or fwrd == "[UNK]":
                    align.append([wrd, buf_o])
                    curr_wrd += 1
                    buf = []
                    buf_o = []
            for item in align:
                word = item[0]
                p_word_list = item[1]
                for p_word in p_word_list:
                    item_list = []
                    item_id_list = []
                    word = word.lower()
                    pretree_list = hownet_dict.get_sememes_by_word(word=word, display='dict')
                    sememe_adj_list = tree2adj(pretree_list)
                    item_list.append([p_word])
                    item_id_list.append(tokenizer.convert_tokens_to_ids([p_word]))
                    senses = []
                    senses_id = []
                    for sememe_adj in sememe_adj_list:
                        sememe = sememe_adj[0]
                        adj = sememe_adj[-1]
                        sememes = []
                        sememes_id = []
                        for item in sememe:
                            item = str(item)
                            if "|||" not in item:
                                sememes_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]))
                            else:
                                sememes_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} |")[1:]))
                            sememes.append(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:])
                        senses.append(sememes)
                        senses_id.append([sememes_id, adj])
                    item_list.append(senses)
                    item_id_list.append(senses_id)
                words_list.append(item_list)
                words_id_list.append(item_id_list)
            sentence_list.append(words_list)
            sentence_id_list.append(words_id_list)
        with open(f"./data/{dataset}/text_sememes_tree_{name}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(sentence_id_list))


if __name__ == "__main__":
    startToken = "[CLS]"
    name = "CoNLL-2003"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    main(name)