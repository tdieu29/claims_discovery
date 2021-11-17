import csv
import pickle
import random
import sys
from functools import partial
from pathlib import Path

sys.path.insert(1, Path(__file__).parent.parent.parent.absolute().__str__())

from colbert.modeling.doc_tokenization import DocTokenizer  # noqa: E402
from colbert.modeling.query_tokenization import QueryTokenizer  # noqa: E402
from colbert.modeling.utils import tensorize_triples  # noqa: E402
from config.config import logger  # noqa: E402


class Batcher:
    def __init__(self, args):
        self.bsize = args.bsize

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )

        self.triples_nf = self._load_triples(args.triples_nf, "NF")
        self.queries_nf = self._load_queries(args.queries_nf, "NF")
        self.collection_nf = self._load_collection(args.collection_nf, "NF")

        self.triples_bioS = self._load_triples(args.triples_bioS, "BioS")
        self.queries_bioS = self._load_queries(args.queries_bioS, "BioS")
        self.collection_bioS = self._load_collection(args.collection_bioS, "BioS")

        self.position_nf, self.position_bioS = 0, 0
        self.stop = False

        self.count = 0
        self.switch_to_bioS = False

    def _load_triples(self, path, name):
        random.seed(12345)

        logger.info(f"#> Loading {name} triples...")

        triples = []

        with open(path) as f:
            csvFile = csv.reader(f)
            next(csvFile)  # Skip first row (header)

        for line in csvFile:
            qid, pos, span_pos, neg, span_neg = line
            qid, pos, span_pos, neg, span_neg = (
                str(qid),
                str(pos),
                int(span_pos),
                str(neg),
                int(span_neg),
            )
            triples.append((qid, pos, span_pos, neg, span_neg))

        # Shuffle triples
        shuffled_triples = random.sample(triples, len(triples))

        return shuffled_triples

    def _load_queries(self, path, name):
        logger.info(f"#> Loading {name} queries...")

        with open(path, "rb") as f:
            queries = pickle.load(f)

        return queries

    def _load_collection(self, path, name):
        logger.info(f"#> Loading {name} collection...")

        with open(path, "rb") as f:
            collection = pickle.load(f)

        return collection

    def __iter__(self):
        return self

    def __next__(self):

        if self.count == 16 and self.switch_to_bioS is False:
            self.switch_to_bioS = True  # Switch to using BioASQ_Scifact
            self.count = 0  # Reset count
        elif self.count == 16 and self.switch_to_bioS is True:
            self.switch_to_bioS = False  # Switch to using NF Corpus
            self.count = 0  # Reset count

        if self.switch_to_bioS is False:
            offset, endpos = self.position_nf, min(
                self.position_nf + self.bsize, len(self.triples_nf)
            )
            self.position_nf = endpos

            triples = self.triples_nf
            queries = self.queries_nf
            collection = self.collection_nf

        elif self.switch_to_bioS is True:
            offset, endpos = self.position_bioS, min(
                self.position_bioS + self.bsize, len(self.triples_bioS)
            )
            self.position_bioS = endpos

            triples = self.triples_bioS
            queries = self.queries_bioS
            collection = self.collection_bioS

        if self.stop is True:
            raise StopIteration

        # Set stop to True once we've reached the end of the triples list
        if self.position_nf == len(self.triples_nf) or self.position_bioS == len(
            self.triples_bioS
        ):  # NEED TO ITERATE OVER BIOS MORE THAN ONCE, SO FIX THIS LINE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.stop = True

        # Retrieve queries, positive examples and negative examples
        queries_list, positives_list, negatives_list = [], [], []

        for position in range(offset, endpos):
            query_id, pos_id, span_pos, neg_id, span_neg = triples[position]

            query = queries[query_id]
            pos = collection[pos_id][span_pos]
            neg = collection[neg_id][span_neg]

            queries_list.append(query)
            positives_list.append(pos)
            negatives_list.append(neg)

        # Increment self.count
        self.count += 1

        batch = self.collate(queries_list, positives_list, negatives_list)

        return batch, self.position_nf, self.position_bioS

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives)

        return self.tensorize_triples(queries, positives, negatives, len(queries) // 2)

    def skip_to_batch(
        self,
        batch_idx,
        intended_batch_size,
        position_nf,
        position_bioS,
        switch_to_bioS,
        count,
    ):
        logger.info(
            f"Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training."
            f"position_nf: {position_nf} | position_bioS: {position_bioS}"
        )

        self.position_bioS = position_bioS
        self.position_nf = position_nf
        self.switch_to_bioS = switch_to_bioS
        self.count = count
