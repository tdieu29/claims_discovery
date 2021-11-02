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
        self.add_argument("--maxsteps", dest="maxsteps", default=23761400, type=int)
        self.add_argument("--bsize", dest="bsize", default=8, type=int)
        self.add_argument("--accum", dest="accumsteps", default=4, type=int)
        self.add_argument("--amp", dest="amp", default=False, action="store_true")

    def add_model_inference_parameters(self):
        self.add_argument("--checkpoint", dest="checkpoint", required=True)
        self.add_argument("--bsize", dest="bsize", default=128, type=int)
        self.add_argument("--amp", dest="amp", default=False, action="store_true")

    def add_training_input(self):
        self.add_argument("--triples", dest="triples", default=None)
        self.add_argument("--queries", dest="queries", default=None)
        self.add_argument("--collection", dest="collection", default=None)
        self.add_argument("--triples_nf", dest="triples_nf")
        self.add_argument("--queries_nf", dest="queries_nf")
        self.add_argument("--collection_nf", dest="collection_nf")
        self.add_argument("--triples_bioS", dest="triples_bioS")
        self.add_argument("--queries_bioS", dest="queries_bioS")
        self.add_argument("--collection_bioS", dest="collection_bioS")
        self.add_argument(
            "--pretrain", dest="pretrain", default=False, action="store_true"
        )
        self.add_argument("--num_epochs", dest="num_epochs", default=1)

        def check_training_input(args):
            if args.triples:
                assert args.queries and args.collection
            elif args.triples_nf:
                assert args.queries_nf and args.collection_nf
            elif args.triples_bioS:
                assert args.queries_bioS and args.collection_bioS

        self.checks.append(check_training_input)

    def add_ranking_input(self):
        self.add_argument("--queries", dest="queries", default=None)
        self.add_argument("--collection", dest="collection", default=None)
        self.add_argument("--qrels", dest="qrels", default=None)

    def add_reranking_input(self):
        self.add_ranking_input()
        self.add_argument("--topk", dest="topK", required=True)
        self.add_argument(
            "--shortcircuit", dest="shortcircuit", default=False, action="store_true"
        )

    def add_indexing_input(self):
        self.add_argument("--collection", dest="collection", required=True)
        self.add_argument("--index_root", dest="index_root", required=True)
        self.add_argument("--index_name", dest="index_name", required=True)

    def add_index_use_input(self):
        self.add_argument("--index_root", dest="index_root", required=True)
        self.add_argument("--index_name", dest="index_name", required=True)
        self.add_argument("--partitions", dest="partitions", default=None, type=int)

    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument("--nprobe", dest="nprobe", default=10, type=int)
        self.add_argument(
            "--retrieve_only", dest="retrieve_only", default=False, action="store_true"
        )

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self):
        args = self.parser.parse_args()
        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        return args
