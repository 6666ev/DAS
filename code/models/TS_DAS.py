import torch.nn as nn
import torch
from utils.tokenizer import LegalTokenizer
import torch.nn.functional as F
from utils.tokenizer import *


class TS_DAS(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, args=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.maps = maps
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if args.pretrain_word_emb:
            tokenizer = LegalTokenizer()
        else:
            tokenizer = MedicalTokenizer()
        vectors = self.load_word_emb(tokenizer)
        self.embedding.weight.data.copy_(vectors)

        self.transformer_enc = nn.Sequential(
            nn.TransformerEncoderLayer(
                self.emb_dim, nhead=10, batch_first=True),
        )
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(emb_dim, len(maps["c2i"])))

        self.fc_ch1 = nn.Linear(emb_dim, emb_dim)
        self.fc_ch2 = nn.Linear(emb_dim, 1)

    def load_word_emb(self):
        tokenizer = LegalTokenizer(
            embedding_path="code/gensim_train/word2vec.model")
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
        x = self.embedding(text)
        hiddens = self.transformer_enc(x)

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
        logits_ch = self.fc_ch2(beta)

        logits_ch = logits_ch.squeeze(-1)

        return {
            "cls": logits_ch,
            "attn_sup": to_sup_alpha,
            "meta": {
                "attn_sup": to_sup_alpha,
                "total_attn": alpha
            }
        }
