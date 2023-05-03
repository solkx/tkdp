import json
import OpenHowNet
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
from tqdm import tqdm
import os
# OpenHowNet.download()



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
                result_list = hownet_dict.get_sense(word)
                sememe_list = []
                for sense_example in result_list:
                    if sense_example.get_sememe_list() not in sememe_list:
                        sememe_list.append(sense_example.get_sememe_list())
                item_list.append([p_word])
                item_id_list.append(tokenizer.convert_tokens_to_ids([p_word]))
                senses = []
                senses_id = []
                for sememe in sememe_list:
                    sememes = []
                    sememes_id = []
                    for item in sememe:
                        item = str(item)
                        sememes_id.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]))
                        sememes.extend(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:])
                    senses.append(sememes)
                    senses_id.append(sememes_id)
                item_list.append(senses)
                item_id_list.append(senses_id)
            words_list.append(item_list)
            words_id_list.append(item_id_list)
        label_list.append(words_list)
        label_id_list.append(words_id_list)
    with open(f"./data/{dataset}/label_sememes_id.json", "w", encoding="utf-8") as f:
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
                    result_list = hownet_dict.get_sense(word)
                    sememe_list = []
                    for sense_example in result_list:
                        if sense_example.get_sememe_list() not in sememe_list:
                            sememe_list.append(sense_example.get_sememe_list())
                    if len(sememe_list) == 0:
                        continue
                    item_list.append([p_word])
                    item_id_list.append(tokenizer.convert_tokens_to_ids([p_word]))
                    senses = []
                    senses_id = []
                    for sememe in sememe_list:
                        sememes = []
                        sememes_id = []
                        for item in sememe:
                            item = str(item)
                            sememes_id.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:]))
                            sememes.extend(tokenizer.tokenize(f"{startToken} {item.split('|')[0]}")[1:])
                        senses.append(sememes)
                        senses_id.append(sememes_id)
                    item_list.append(senses)
                    item_id_list.append(senses_id)
                words_list.append(item_list)
                words_id_list.append(item_id_list)
            sentence_list.append(words_list)
            sentence_id_list.append(words_id_list)
        with open(f"./data/{dataset}/text_sememes_id_{name}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(sentence_id_list))


if __name__ == "__main__":
    startToken = "[CLS]"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for name in ["CoNLL-2003"]:
        main(name)