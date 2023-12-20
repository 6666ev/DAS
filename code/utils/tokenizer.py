from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import torch

class LegalTokenizer:
    def __init__(self, embedding_path="code/gensim_train/word2vec.model") -> None:
        model = Word2Vec.load(embedding_path)
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.id2word = self.special_tokens + model.wv.index_to_key

        self.word2id = model.wv.key_to_index  
        for k in self.word2id.keys():
            self.word2id[k] += len(self.special_tokens)
        for i in range(len(self.special_tokens)):
            self.word2id[self.special_tokens[i]] = i
        self.embedding_path = embedding_path
        self.vector_size = model.wv.vector_size
        self.vocab_size = len(self.word2id)
        special_token_vec = np.zeros(
            (len(self.special_tokens), self.vector_size))
        self.vectors = model.wv.vectors  
        self.vectors = np.concatenate(
            (special_token_vec, self.vectors))  

    def load_embedding(self):
        return self.vectors

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, sents, max_length=512, return_tensors="ls", padding="max_length", truncation=True):
        
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for sent in sents:
            sent = sent.split(" ")
            sent = [self.word2id[w] if w in self.word2id.keys() else self.word2id["[UNK]"] for w in sent]
            sent = [self.word2id["[SOS]"]] + sent + [self.word2id["[EOS]"]]
            sent_len = len(sent)
            sent += [0] * max_length
            sent = sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([0] * max_length)
            mask = [1] * sent_len + [0] * max_length
            attention_mask.append(mask[:max_length])

        if return_tensors == "np":
            input_ids = np.array(input_ids)
            token_type_ids = np.array(token_type_ids)
            attention_mask = np.array(attention_mask)
        elif return_tensors == 'pt':
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token):
        sent = []
        for t in token:
            if t == 0:
                break
            if t > len(self.id2word):
                sent.append("[UNK]")
            else:
                sent.append(self.id2word[t])
        sent = " ".join(sent)
        return sent

class MedicalTokenizer:
    def __init__(self, embedding_path="code/gensim_train/medical/word2vec.model") -> None:
        model = Word2Vec.load(embedding_path)
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.id2word = self.special_tokens + model.wv.index_to_key

        self.word2id = model.wv.key_to_index  
        for k in self.word2id.keys():
            self.word2id[k] += len(self.special_tokens)
        for i in range(len(self.special_tokens)):
            self.word2id[self.special_tokens[i]] = i
        self.embedding_path = embedding_path
        self.vector_size = model.wv.vector_size
        self.vocab_size = len(self.word2id)
        special_token_vec = np.zeros(
            (len(self.special_tokens), self.vector_size))
        self.vectors = model.wv.vectors  
        self.vectors = np.concatenate(
            (special_token_vec, self.vectors))  

    def load_embedding(self):
        return self.vectors

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, sents, max_length=512, return_tensors="ls", padding="max_length", truncation=True):
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for sent in sents:
            sent = sent.split(" ")
            sent = [self.word2id[w] if w in self.word2id.keys() else self.word2id["[UNK]"] for w in sent]
            sent = [self.word2id["[SOS]"]] + sent + [self.word2id["[EOS]"]]
            sent_len = len(sent)
            sent += [0] * max_length
            sent = sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([0] * max_length)
            mask = [1] * sent_len + [0] * max_length
            attention_mask.append(mask[:max_length])

        if return_tensors == "np":
            input_ids = np.array(input_ids)
            token_type_ids = np.array(token_type_ids)
            attention_mask = np.array(attention_mask)
        elif return_tensors == 'pt':
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token):
        sent = []
        for t in token:
            if t == 0:
                break
            if t > len(self.id2word):
                sent.append("[UNK]")
            else:
                sent.append(self.id2word[t])
        sent = " ".join(sent)
        return sent


