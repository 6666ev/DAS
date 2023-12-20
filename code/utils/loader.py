import jieba
import re
import os
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
from tqdm import tqdm
import numpy as np
import json

RANDOM_SEED = 22
torch.manual_seed(RANDOM_SEED)


def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("code/utils/stopword.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    # text = re.sub("[0-9\.]+元", "", text)
    # stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    stop_words_user = []
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]

    text = " ".join(text)
    text = re.sub(r'\s+', ' ', text)
    return text


class CailDataset(Dataset):
    def __init__(self, dataset_name, facts, charges, attn_sup):
        # super().__init__()
        self.facts = facts
        self.charges = torch.LongTensor(charges)
        self.attn_sup = torch.Tensor(attn_sup)

    def __getitem__(self, idx):
        return {
            "text":
                {
                    "input_ids": self.facts["input_ids"][idx],
                    "token_type_ids": self.facts["token_type_ids"][idx],
                    "attention_mask": self.facts["attention_mask"][idx],
                },
            "cls": self.charges[idx],
            "attn_sup": self.attn_sup[idx]
        }

    def __len__(self):
        return len(self.facts["input_ids"])



def label2idx(label, map):
    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j] = map[str(label[i][j])]

    return label


def gau(x):
    x2 = x**2
    return np.exp(-x2 / 2)


def get_one_charge_att_sup(kws, tokens, dataset_name, tokenizer, seq_len):
    attn = [0] * 1024

    text = tokenizer.decode(tokens)
    text = text.split()
    for idx, w in enumerate(text):
        if w in kws:
            attn[idx] = 1

    attn = attn[:seq_len]

    # attention smooth
    # attn_star = [0] * len(attn)
    # for i in range(len(attn)):
    #     if attn[i]:
    #         for j in range(i - 3, i + 3):
    #             if j < 0 or j >= len(attn):
    #                 continue
    #             attn_star[j] += gau(i-j)
                
    # for i in range(len(attn_star)):
    #     attn_star[i] = min(1, attn_star[i])
    # attn = attn_star

    return attn



def load_cail_data(filepath, dataset_name, tokenizer, kw_path):
    facts, charges = [], []
    with open(filepath) as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            ch = json_obj["meta"]["accusation"]

            facts.append(json_obj["fact"])
            charges.append(ch)
            if len(json_obj["fact"].split()) < 10:
                    continue
    attn_sup = []

    kw_path = "data/filter/cail/total/meta/kw_task_p100.json"
    c2kw = json.load(open(kw_path))

    pkl_path = "code/pkl/{}/train_clean.pkl".format(dataset_name)
    if not os.path.exists(pkl_path):
        path, _ = os.path.split(pkl_path)
        if not os.path.exists(path):
            os.makedirs(path)

        facts = tokenizer(facts, max_length=512, return_tensors="pt", padding="max_length", truncation=True)

        with open(pkl_path, "wb") as f:
            pickle.dump(facts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl data saved: {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        facts = pickle.load(f)

    # 获取attention supervision
    for tokens, ch in tqdm(zip(facts["input_ids"], charges)):
        cur_attn_sup = []
        kws_merge = set([kw for c in ch for kw in c2kw[c]])
        cur_attn_sup = get_one_charge_att_sup(kws_merge, tokens, dataset_name, tokenizer, seq_len=512)
        
        attn_sup.append(cur_attn_sup)

    ret_maps = {}

    c2i_path = "data/filter/cail/total/meta/c2i.json" # single label

    with open(c2i_path) as f:
        c2i = json.load(f)
        ret_maps["c2i"] = c2i
        ret_maps["i2c"] = {v: k for k, v in c2i.items()}

    charges = label2idx(charges, ret_maps["c2i"])

    charges = [c[0] for c in charges]
    dataset = CailDataset(dataset_name, facts, charges, attn_sup)

    return dataset, ret_maps


class MedicalDataset(Dataset):
    def __init__(self, dataset_name, texts, departments, attn_sup):
        # super().__init__()
        self.texts = texts
        self.departments = torch.LongTensor(departments)
        self.attn_sup = torch.Tensor(attn_sup)

    def __getitem__(self, idx):
        return {
            "text":
                {
                    "input_ids": self.texts["input_ids"][idx],
                    "token_type_ids": self.texts["token_type_ids"][idx],
                    "attention_mask": self.texts["attention_mask"][idx],
                },
            "cls": self.departments[idx],
            "attn_sup": self.attn_sup[idx]
        }

    def __len__(self):
        return len(self.texts["input_ids"])


def load_medical_data(filepath, dataset_name, tokenizer, kw_path):
    seq_len = 64
    texts, departments = [], []
    with open(filepath) as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            ask = json_obj["ask"]
            text = f"{ask}"
            if len(text.split()) < 10:
                continue
                
            texts.append(text)
            departments.append(json_obj["department"])

    pkl_path = "code/pkl/{}/train_clean.pkl".format(dataset_name)
    if not os.path.exists(pkl_path):
        path, _ = os.path.split(pkl_path)
        if not os.path.exists(path):
            os.makedirs(path)

        texts = tokenizer(texts, max_length=seq_len, return_tensors="pt", padding="max_length", truncation=True)

        with open(pkl_path, "wb") as f:
            pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl data saved: {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        texts = pickle.load(f)

    # 获取attention supervision
    kw_path = "data/medical/filter/total/meta/kw_task_p100.json"

    c2kw = json.load(open(kw_path))
    attn_sup = []
    for tokens, department in tqdm(zip(texts["input_ids"], departments)):
        kws = c2kw[department]
        cur_attn_sup = get_one_charge_att_sup(kws, tokens, dataset_name, tokenizer, seq_len)
        attn_sup.append(cur_attn_sup)

    ret_maps = {}

    c2i_path = "data/medical/filter/total/meta/d2i.json"

    with open(c2i_path) as f:
        c2i = json.load(f)
        ret_maps["c2i"] = c2i
        ret_maps["i2c"] = {v: k for k, v in c2i.items()}

    departments = [[i] for i in departments]
    departments = label2idx(departments, ret_maps["c2i"])
    departments = [i[0] for i in departments]

    dataset = MedicalDataset(dataset_name, texts, departments, attn_sup)

    return dataset, ret_maps

