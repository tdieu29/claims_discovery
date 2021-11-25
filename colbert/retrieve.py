import json
import math
import os
import random
from typing import OrderedDict

from colbert.indexing.faiss import get_faiss_index_name
from colbert.indexing.loaders import load_doclens
from colbert.parameters import retrieval_params
from colbert.ranking.retrieval import retrieve
from colbert.utils.parser import Arguments
from colbert.utils.utils import load_colbert


def retrieve_abstracts(query: str, colbert=None, checkpoint=None):
    random.seed(12345)

    parser = Arguments(description="End-to-end retrieval and ranking with ColBERT.")

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_ranking_input()
    parser.add_retrieval_input()

    params_list = retrieval_params
    args = parser.parse(params_list)

    if colbert is None and checkpoint is None:
        args.colbert, args.checkpoint = load_colbert(args)
    else:
        args.colbert, args.checkpoint = colbert, checkpoint

    args.queries = query

    args.index_path = os.path.join(args.index_root, args.index_name)

    fp = os.path.join(args.index_path, "num_embeddings/num_embeddings.json")
    with open(fp) as file:
        args.num_embeddings = json.load(file)

    if args.faiss_name is not None:
        args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
        fname_components = os.path.basename(args.faiss_index_path).split(".")
        args.partitions = int(fname_components[1])

        # Retrieve relevant abstracts
        retrieve(args)
    else:
        step = math.ceil(args.num_faiss_indexes / args.slices)
        for count in range(args.num_faiss_indexes):
            num_embeddings = sum(load_doclens(args.index_path, count, step))
            args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))

            args.faiss_index_path = os.path.join(
                args.index_path, get_faiss_index_name(args, count, count + step)
            )

        # Retrieve relevant abstracts
        retrieve(args)

    # Copy all of the retrieved abstracts into one dictionary
    all_abstracts = OrderedDict()
    file_paths = [
        os.path.join(
            args.index_path,
            "abstracts_retrieved",
            f"abstracts_retrieved_{i}-{i+1}.json",
        )
        for i in range(args.num_faiss_indexes)
    ]
    for fp in file_paths:
        with open(fp) as file:
            retrieved_abstracts = json.load(file)
            for article_id in retrieved_abstracts:
                assert article_id not in all_abstracts
                all_abstracts[article_id] = retrieved_abstracts[article_id]

    # Sort abstracts by scores (highest to lowest scores)
    all_abstracts_sorted = sorted(
        all_abstracts.items(), key=lambda x: x[1], reverse=True
    )
    all_abstracts_sorted = OrderedDict(all_abstracts_sorted[: args.final_num_abstracts])

    # Save sorted abstracts to file
    fp = os.path.join(
        args.index_path, "abstracts_retrieved", "abstracts_retrieved_final.json"
    )
    with open(fp, "w") as file:
        json.dump(all_abstracts_sorted, file)

    return all_abstracts_sorted
