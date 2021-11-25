import os
import random

from colbert.indexing.encoder import CollectionEncoder
from colbert.parameters import index_params
from colbert.utils.parser import Arguments
from colbert.utils.utils import create_directory


def main():
    random.seed(12345)

    parser = Arguments()
    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()

    params_list = index_params()
    args = parser.parse(params_list)

    args.index_path = os.path.join(args.index_root, args.index_name)
    assert not os.path.exists(args.index_path), args.index_path

    create_directory(args.index_root)
    create_directory(args.index_path)

    encoder = CollectionEncoder(args)
    encoder.encode()


if __name__ == "__main__":
    main()
