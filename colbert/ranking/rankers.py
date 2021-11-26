from functools import partial

import torch

from colbert.ranking.faiss_index import FaissIndex
from colbert.ranking.index_part import IndexPart


class Ranker:
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(
                args.num_embeddings, args.index_path, args.faiss_index_path, args.nprobe
            )
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(
            args.num_embeddings,
            args.index_path,
            inference.colbert.dim,
            self.faiss_index.faiss_part_range,
            verbose=True,
        )

    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(
            queries, bsize=512 if len(queries) > 512 else None
        )

        return Q

    def rank(self, Q, pids=None):
        pids = self.retrieve(Q, verbose=False)[0] if pids is None else pids

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)  # (number of queries, dim, query_seq_len)
            scores = self.index.rank(Q, pids)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            pids, scores = (
                torch.tensor(pids)[scores_sorter.indices].tolist(),
                scores_sorter.values.tolist(),
            )

        return pids, scores
