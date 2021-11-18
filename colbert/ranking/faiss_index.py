import os

import faiss
import torch

from colbert.indexing.loaders import load_doclens
from colbert.utils.utils import flatten
from config.config import logger


class FaissIndex:
    def __init__(self, index_path, faiss_index_path, nprobe, part_range=None):
        logger.info(f"#> Loading the FAISS index from {faiss_index_path} ...")

        faiss_part_range = os.path.basename(faiss_index_path).split(".")
        if len(faiss_part_range) <= 3:
            faiss_part_range = None
        else:
            faiss_part_range = faiss_part_range[-2].split("-")

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))
            assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
            assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)

        self.part_range = part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index.nprobe = nprobe

        logger.info("#> Building the emb2pid mapping...")
        all_doclens = load_doclens(index_path, flatten=False)

        pid_offset = 0
        if faiss_part_range is not None:
            logger.info(f"#> Restricting all_doclens to the range {faiss_part_range}.")
            pid_offset = len(flatten(all_doclens[: faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start : faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = (
                self.faiss_part_range.start if self.faiss_part_range is not None else 0
            )
            a = len(flatten(all_doclens[: self.part_range.start - start]))
            b = len(flatten(all_doclens[: self.part_range.stop - start]))
            self.relative_range = range(a, b)
            logger.info(f"self.relative_range = {self.relative_range}")
        if os.path.exists(os.path.join(index_path, "emb2pid", "emb2pid.pt")):
            logger.info("#> Retrieving the emb2pid mapping...")
            self.emb2pid = torch.load(os.path.join(index_path, "emb2pid", "emb2pid.pt"))
        else:
            logger.info("#> Building the emb2pid mapping..")
            os.makedirs(os.path.join(index_path, "emb2pid"), exist_ok=True)

            all_doclens = flatten(all_doclens)

            total_num_embeddings = sum(all_doclens)
            self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

            offset_doclens = 0
            for pid, dlength in enumerate(all_doclens):
                self.emb2pid[offset_doclens : offset_doclens + dlength] = (
                    pid_offset + pid
                )
                offset_doclens += dlength

            # Save emb2pid mapping
            torch.save(self.emb2pid, os.path.join(index_path, "emb2pid.pt"))

        logger.info(f"len(self.emb2pid) = {len(self.emb2pid)}")

    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        if self.relative_range is not None:
            pids = [
                [pid for pid in pids_ if pid in self.relative_range] for pids_ in pids
            ]

        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        logger.info(
            "#> Search in batches with faiss. \t\t"
            f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}"
        )

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            logger.info(f"#> Searching from {offset} to {endpos}...")

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            _, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(
            num_queries, embeddings_per_query * embedding_ids.size(1)
        )

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):

        # Find unique PIDs per query.
        logger.info("#> Lookup the PIDs..")
        all_pids = self.emb2pid[embedding_ids]

        logger.info(f"#> Converting to a list [shape = {all_pids.size()}]..")
        all_pids = all_pids.tolist()

        logger.info("#> Removing duplicates (in parallel if large enough)..")

        if len(all_pids) > 5000:
            all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq, all_pids))

        logger.info("#> Done with embedding_ids_to_pids().")

        return all_pids


def uniq(l):
    return list(set(l))
