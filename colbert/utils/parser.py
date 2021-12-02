import copy
from argparse import ArgumentParser


class Arguments:
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = []

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument(
            "--similarity", dest="similarity", default="l2", choices=["cosine", "l2"]
        )
        self.add_argument("--dim", dest="dim", default=128, type=int)
        self.add_argument("--query_maxlen", dest="query_maxlen", default=128, type=int)
        self.add_argument("--doc_maxlen", dest="doc_maxlen", default=512, type=int)

        # Filtering-related Arguments
        self.add_argument(
            "--mask-punctuation",
            dest="mask_punctuation",
            default=False,
            action="store_true",
        )

    def add_model_training_parameters(self):
        # NOTE: Providing a checkpoint is one thing, --resume is another, --resume_optimizer is yet another.
        self.add_argument("--resume", dest="resume", default=False, action="store_true")
        self.add_argument(
            "--resume_optimizer",
            dest="resume_optimizer",
            default=False,
            action="store_true",
        )
        self.add_argument(
            "--checkpoint", dest="checkpoint", default=None, required=False
        )

        self.add_argument("--lr", dest="lr", default=2e-05, type=float)
        self.add_argument("--maxsteps", dest="maxsteps", default=24000000, type=int)
        self.add_argument("--bsize", dest="bsize", default=8, type=int)
        self.add_argument("--accum", dest="accumsteps", default=4, type=int)
        self.add_argument("--amp", dest="amp", default=False, action="store_true")

    def add_model_inference_parameters(self):
        self.add_argument("--checkpoint", dest="checkpoint", required=True)
        self.add_argument("--bsize", dest="bsize", default=256, type=int)
        self.add_argument("--amp", dest="amp", default=False, action="store_true")

    def add_training_input(self):
        self.add_argument("--triples", dest="triples", required=False)
        self.add_argument("--queries", dest="queries", required=False)
        self.add_argument("--collection", dest="collection", required=False)
        self.add_argument("--triples_nf", dest="triples_nf", required=False)
        self.add_argument("--queries_nf", dest="queries_nf", required=False)
        self.add_argument("--collection_nf", dest="collection_nf", required=False)
        self.add_argument("--triples_bioS", dest="triples_bioS", required=False)
        self.add_argument("--queries_bioS", dest="queries_bioS", required=False)
        self.add_argument("--collection_bioS", dest="collection_bioS", required=False)
        self.add_argument(
            "--pretrain", dest="pretrain", default=False, action="store_true"
        )
        self.add_argument("--num_epochs", dest="num_epochs", default=1)

        def check_training_input(args):
            if args.triples:
                assert args.queries and args.collection
            if args.triples_nf:
                assert args.queries_nf and args.collection_nf
            if args.triples_bioS:
                assert args.queries_bioS and args.collection_bioS
            if not args.triples:
                assert args.triples_nf and args.triples_bioS

        self.checks.append(check_training_input)

    def add_ranking_input(self):
        self.add_argument("--queries", dest="queries")

    def add_indexing_input(self):
        self.add_argument(
            "--collection",
            dest="collection",
            default="cord19_data/database/articles.sqlite",
        )
        self.add_argument(
            "--index_root", dest="index_root", default="colbert/faiss_indexes"
        )
        self.add_argument("--index_name", dest="index_name", required=True)
        self.add_argument(
            "--chunksize", dest="chunksize", default=6.0, type=float
        )  # in GiBs

    def add_index_use_input(self):
        self.add_argument(
            "--index_root", dest="index_root", default="colbert/faiss_indexes"
        )
        self.add_argument("--index_name", dest="index_name", required=True)
        self.add_argument("--partitions", dest="partitions", default=None, type=int)
        self.add_argument("--sample", dest="sample", default=0.3, type=float)
        self.add_argument("--slices", dest="slices", type=int)

    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument("--nprobe", dest="nprobe", default=64, type=int)
        self.add_argument("--faiss_name", dest="faiss_name", default=None, type=str)
        self.add_argument("--faiss_depth", dest="faiss_depth", default=1024, type=int)
        self.add_argument("--num_faiss_indexes", dest="num_faiss_indexes", type=int)
        self.add_argument(
            "--num_retrieved_abstracts",
            dest="num_retrieved_abstracts",
            default=20,
            type=int,
        )
        self.add_argument(
            "--final_num_abstracts", dest="final_num_abstracts", default=100, type=int
        )

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self, arguments=None):
        args = self.parser.parse_args(arguments)
        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        return args
