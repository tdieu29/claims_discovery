import os
import random

from colbert.indexing.faiss import index_faiss
from colbert.parameters import index_faiss_params
from colbert.utils.parser import Arguments


def main():
    random.seed(12345)

    parser = Arguments(
        description="Faiss indexing for end-to-end retrieval with ColBERT."
    )
    parser.add_index_use_input()

    params_list = index_faiss_params()
    args = parser.parse(params_list)

    assert args.slices >= 1
    assert args.sample is None or (0.0 < args.sample <= 1.0), args.sample

    args.index_path = os.path.join(args.index_root, args.index_name)
    assert os.path.exists(args.index_path), args.index_path

    index_faiss(args)


if __name__ == "__main__":
    main()
