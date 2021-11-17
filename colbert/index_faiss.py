import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(1, Path(__file__).parent.parent.absolute().__str__())

from colbert.indexing.faiss import index_faiss  # noqa: E402
from colbert.indexing.loaders import load_doclens  # noqa: E402
from colbert.utils.parser import Arguments  # noqa: E402
from config.config import logger  # noqa: E402


def main():
    random.seed(12345)

    parser = Arguments(
        description="Faiss indexing for end-to-end retrieval with ColBERT."
    )
    parser.add_index_use_input()

    parser.add_argument("--sample", dest="sample", default=1.0, type=float)
    parser.add_argument("--slices", dest="slices", default=1, type=int)

    args = parser.parse()
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
