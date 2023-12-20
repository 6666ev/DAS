import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tokenizer import *


class BiLSTM_DAS(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, args=None):
        super(BiLSTM_DAS, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.maps = maps
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if args.pretrain_word_emb:
            tokenizer = LegalTokenizer()
        else:
            tokenizer = MedicalTokenizer()
        vectors = self.load_word_emb(tokenizer)
        self.embedding.weight.data.copy_(vectors)

        self.lstm = nn.LSTM(emb_dim, hid_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.4)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hid_dim * 2, len(maps["c2i"])))

        self.fc_ch1 = nn.Linear(2 * hid_dim, hid_dim)
        self.fc_ch2 = nn.Linear(hid_dim, 1)

    def load_word_emb(self, tokenizer):
        vectors = tokenizer.load_embedding()
        vectors = torch.Tensor(vectors)
        return vectors

    def debias_attention_mean(self, alpha):
        tau = 0.05
        mean = torch.mean(alpha, dim=-1).unsqueeze(-1)
        alpha = alpha - mean
        n = alpha.shape[-1]

        alpha = alpha / tau
        alpha = F.softmax(alpha, dim=1)
        return alpha

    def forward(self, data):
        text = data["text"]["input_ids"].cuda()

        emb = self.embedding(text)
        hiddens, _ = self.lstm(emb)

        hiddens = self.tanh1(hiddens)
        alpha_logits = torch.matmul(hiddens, self.w)

        charge_idx = data["cls"].unsqueeze(
            1).repeat(1, text.shape[1]).unsqueeze(-1)

        mask = data["text"]["attention_mask"].cuda().float() - 1
        mask *= 1e9
        alpha_logits += mask.unsqueeze(-1)

        alpha = F.softmax(alpha_logits, dim=1)
        to_sup_alpha = alpha.gather(-1, charge_idx).squeeze(-1)

        alpha = self.debias_attention_mean(alpha)

        alpha = alpha.transpose(1, 2)

        beta = torch.bmm(alpha, hiddens)
        logits_ch = self.fc_ch2(nn.ReLU()(self.fc_ch1(beta)))
        logits_ch = logits_ch.squeeze(-1)

        return {
            "cls": logits_ch,
            "attn_sup": to_sup_alpha,
            "meta": {
                "attn_sup": to_sup_alpha,
                "total_attn": alpha
            }
        }
