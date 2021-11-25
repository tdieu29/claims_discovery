import math
import os
import random

from colbert.indexing.faiss import index_faiss
from colbert.indexing.loaders import load_doclens
from colbert.parameters import index_use_params
from colbert.utils.parser import Arguments
from config.config import logger


def main():
    random.seed(12345)

    parser = Arguments(
        description="Faiss indexing for end-to-end retrieval with ColBERT."
    )
    parser.add_index_use_input()

    params_list = index_use_params()
    args = parser.parse(params_list)

    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample <= 1.0), args.sample

    args.index_path = os.path.join(args.index_root, args.index_name)
    assert os.path.exists(args.index_path), args.index_path

    num_embeddings = sum(load_doclens(args.index_path))
    logger.info(f"#> num_embeddings = {num_embeddings}")

    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        logger.info(
            "You did not specify --partitions!\n"
            f"Default computation chooses {args.partitions} partitions"
            f"(for {num_embeddings} embeddings)"
        )

    index_faiss(args)


if __name__ == "__main__":
    main()
