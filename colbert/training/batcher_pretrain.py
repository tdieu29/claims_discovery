import csv
import pickle
import random
from functools import partial

from colbert.modeling.doc_tokenization import DocTokenizer
from colbert.modeling.query_tokenization import QueryTokenizer
from colbert.modeling.utils import tensorize_triples
from config.config import logger


class Batcher_Pretrain:
    def __init__(self, args):
        self.bsize = args.bsize

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )

        self.triples = self._load_triples(args.triples)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)

        self.position = 0
        self.stop = False

    def _load_triples(self, path):
        random.seed(12345)

        logger.info("#> Loading triples...")

        triples = []

        with open(path) as f:
            csvFile = csv.reader(f)
            next(csvFile)  # Skip first row (header)

            for line in csvFile:
                qid, pos, neg, _ = line
                triples.append((qid, pos, neg))

        # Shuffle triples
        shuffled_triples = random.sample(triples, len(triples))

        return shuffled_triples

    def _load_queries(self, path):
        logger.info("#> Loading queries...")

        with open(path, "rb") as f:
            queries = pickle.load(f)

        return queries

    def _load_collection(self, path):
        logger.info("#> Loading collection...")

        with open(path, "rb") as f:
            collection = pickle.load(f)

        return collection

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(
            self.position + self.bsize, len(self.triples)
        )
        self.position = endpos

        if self.stop is True:
            raise StopIteration

        # Set stop to True once we've reached the end of the triples tuple
        if self.position == len(self.triples):
            self.stop = True

        queries, positives, negatives = [], [], []

        for position in range(offset, endpos):
            query_id, pos_id, neg_id = self.triples[position]
            query, pos, neg = (
                self.queries[int(query_id)],
                self.collection[int(pos_id)],
                self.collection[int(neg_id)],
            )

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives)

        return self.tensorize_triples(queries, positives, negatives, len(queries) // 2)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        logger.info(
            f"Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training."
        )
        self.position = intended_batch_size * batch_idx
