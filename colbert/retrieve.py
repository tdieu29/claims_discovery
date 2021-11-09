import os
import random

from colbert.indexing.faiss import get_faiss_index_name
from colbert.ranking.retrieval import retrieve
from colbert.utils.parser import Arguments
from colbert.utils.utils import load_colbert


def retrieve_abstracts(query: str, colbert, checkpoint):
    random.seed(12345)

    parser = Arguments(description="End-to-end retrieval and ranking with ColBERT.")

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_ranking_input()
    parser.add_retrieval_input()

    parser.add_argument("--faiss_name", dest="faiss_name", default=None, type=str)
    parser.add_argument("--faiss_depth", dest="faiss_depth", default=1024, type=int)
    parser.add_argument("--part-range", dest="part_range", default=None, type=str)
    parser.add_argument("--depth", dest="depth", default=1024, type=int)

    args = parser.parse()

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split(".."))
        args.part_range = range(part_offset, part_endpos)

    args.colbert, args.checkpoint = load_colbert(args)
    args.colbert, args.checkpoint = colbert, checkpoint
    args.queries = query

    args.index_path = os.path.join(args.index_root, args.index_name)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
    else:
        args.faiss_index_path = os.path.join(
            args.index_path, get_faiss_index_name(args)
        )

    abstracts_retrieved = retrieve(args)

    return abstracts_retrieved
