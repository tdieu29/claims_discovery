import torch

from colbert.modeling.colbert import ColBERT
from colbert.modeling.doc_tokenization import DocTokenizer
from colbert.modeling.query_tokenization import QueryTokenizer
from colbert.utils.amp import MixedPrecisionManager


class ModelInference:
    def __init__(self, colbert: ColBERT, amp=False):
        assert colbert.training is False

        self.colbert = colbert
        self.query_tokenizer = QueryTokenizer(colbert.query_maxlen)
        self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)
                return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batches = [
                self.query(input_ids, attention_mask, to_cpu=to_cpu)
                for input_ids, attention_mask in batches
            ]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            batches, original_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                for input_ids, attention_mask in batches
            ]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[original_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in original_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)


def _stack_3D_tensors(groups):
    bsize = sum(x.size(0) for x in groups)
    maxlen = max(x.size(1) for x in groups)
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output
