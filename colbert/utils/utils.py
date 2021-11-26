import itertools
import os

import torch

from colbert.modeling.colbert import ColBERT
from colbert.parameters import DEVICE
from config.config import logger


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_checkpoint(path, model, optimizer=None, do_log=True):
    if do_log:
        logger.info(f"#> Loading checkpoint {path} ...")

    checkpoint = torch.load(path, map_location="cpu")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        logger.warning("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if do_log:
        logger.info(f"#> checkpoint['epoch'] = {checkpoint['epoch']}")
        logger.info(f"#> checkpoint['batch'] = {checkpoint['batch']}")

    return checkpoint


def load_colbert(args, do_log=True):
    colbert = ColBERT(
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )
    colbert = colbert.to(DEVICE)

    if do_log:
        logger.info("#> Loading model checkpoint.")

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_log=do_log)

    colbert.eval()

    return colbert


def create_directory(path):
    if os.path.exists(path):
        logger.info(f"#> Note: Output directory {path} already exists.")
    else:
        logger.info(f"#> Creating directory {path}")
        os.makedirs(path)


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset : offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


def flatten(L):
    return [x for y in L for x in y]
