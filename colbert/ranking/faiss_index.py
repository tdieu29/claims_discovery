import os

import faiss
import torch

from colbert.indexing.loaders import load_doclens
from colbert.utils.utils import flatten
from config.config import logger


class FaissIndex:
    def __init__(self, num_embeddings_dict, index_path, faiss_index_path, nprobe):
        logger.info(f"#> Loading the FAISS index from {faiss_index_path} ...")

        faiss_part_range = os.path.basename(faiss_index_path).split(".")
        if len(faiss_part_range) <= 3:
            faiss_part_range = None
        else:
            faiss_part_range = faiss_part_range[-2].split("-")

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))

        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index.nprobe = nprobe

        all_doclens = load_doclens(index_path, flatten=False)

        self.pids_range = None
        pid_offset = 0
        if faiss_part_range is not None:
            logger.info(f"#> Restricting all_doclens to the range {faiss_part_range}.")

            pid_offset = len(flatten(all_doclens[: faiss_part_range.start]))
            pid_endpos = len(flatten(all_doclens[: faiss_part_range.stop]))
            self.pids_range = range(pid_offset, pid_endpos)

            all_doclens = all_doclens[faiss_part_range.start : faiss_part_range.stop]

        # Retrieve or build emb2pid mapping
        fp = os.path.join(
            index_path,
            "emb2pid",
            f"emb2pid_{faiss_part_range.start}-{faiss_part_range.stop}.pt",
        )
        if os.path.exists(fp):
            logger.info("#> Retrieving the emb2pid mapping...")
            self.emb2pid = torch.load(fp)
        else:
            logger.info("#> Building the emb2pid mapping..")
            dir_path = os.path.join(index_path, "emb2pid")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            all_doclens = flatten(all_doclens)

            total_num_embeddings = sum(all_doclens)
            assert (
                total_num_embeddings == num_embeddings_dict[str(faiss_part_range.start)]
            )
            self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

            offset_doclens = 0
            for pid, dlength in enumerate(all_doclens):
                self.emb2pid[offset_doclens : offset_doclens + dlength] = (
                    pid_offset + pid
                )
                offset_doclens += dlength

            # Save emb2pid mapping
            torch.save(self.emb2pid, fp)

        logger.info(
            f"len(self.emb2pid) (faiss range: {faiss_part_range.start}-{faiss_part_range.stop}) = {len(self.emb2pid)}"
        )

    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        if self.pids_range is not None:
            for pids_ in pids:
                for pid in pids_:
                    assert pid in self.pids_range

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
