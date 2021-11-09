import string

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from colbert.parameters import DEVICE


class ColBERT(nn.Module):
    def __init__(
        self, query_maxlen, doc_maxlen, mask_punctuation, dim, similarity_metric
    ):

        super().__init__()

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim
        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "monologg/biobert_v1.1_pubmed"
            )
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }

        self.bert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.linear = nn.Linear(768, dim, bias=False)

    def forward(self, Q, D, doc_attn_mask):
        return self.score(self.query(*Q), self.doc(*D), doc_attn_mask)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D, doc_attn_mask):
        if self.similarity_metric == "cosine":
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == "l2"
        score = -1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)
        bool_mask = doc_attn_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~bool_mask] = -100000
        score = score.max(-1).values.sum(-1)
        return score

    def mask(self, input_ids):
        mask = [
            [(x not in self.skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask
