import torch

from colbert.indexing.index_manager import load_index_part
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.ranking.index_ranker import IndexRanker
from colbert.utils.utils import flatten
from config.config import logger


class IndexPart:
    def __init__(self, directory, dim=128, part_range=None, verbose=True):
        first_part, last_part = (
            (0, None) if part_range is None else (part_range.start, part_range.stop)
        )

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum(
            len(part_doclens) for part_doclens in all_doclens[:first_part]
        )
        self.doc_endpos = sum(
            len(part_doclens) for part_doclens in all_doclens[:last_part]
        )
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.tensor = self._load_parts(dim, verbose)
        self.ranker = IndexRanker(self.tensor, self.doclens)

    def _load_parts(self, dim, verbose):
        """
        Load all of the saved embeddings into a 2D matrix.

        Args:
            dim ([type]): [description]
            verbose ([type]): [description]

        Returns:
            [type]: [description]
        """
        tensor = torch.zeros(self.num_embeddings + 512, dim, dtype=torch.float16)

        if verbose:
            logger.info(f"tensor.size() = {tensor.size()}")

        offset = 0
        for idx, filename in enumerate(self.parts_paths):
            logger.info(f"|> Loading {filename} ...")

            endpos = offset + sum(self.parts_doclens[idx])
            part = load_index_part(filename, verbose=verbose)

            tensor[offset:endpos] = part
            offset = endpos

        return tensor

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores
