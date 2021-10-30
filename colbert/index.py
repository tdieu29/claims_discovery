import os
import random

from colbert.utils.utils import create_directory
from colbert.indexing.encoder import CollectionEncoder


def main():
    random.seed(12345)

    #parser = Arguments()
    #parser.add_model_parameters()
    #parser.add_model_inference_parameters()
    #parser.add_indexing_input()

    #parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)   # in GiBs

    #args = parser.parse()

    args.index_path = os.path.join(args.index_root, args.index_name)
    assert not os.path.exists(args.index_path), args.index_path

    create_directory(args.index_root)
    create_directory(args.index_path)

    encoder = CollectionEncoder(args)
    encoder.encode()

if __name__ == "__main__":
    main()
