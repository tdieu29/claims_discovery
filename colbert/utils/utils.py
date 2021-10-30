import os
import torch
import itertools 
import datetime 

from colbert.modeling.colbert import ColBERT
from colbert.parameters import DEVICE

def print_message(*s, condition=True): 
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg

def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def load_checkpoint(path, model, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path, "...")
  
    checkpoint = torch.load(path, map_location='cpu')

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except: 
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  
    if optimizer: 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  
    if do_print: 
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])
  
    return checkpoint

def load_colbert(args, do_print=True):
    colbert = ColBERT(query_maxlen = args.query_maxlen,
                    doc_maxlen = args.doc_maxlen,
                    dim = args.dim,
                    similarity_metric = args.similarity,
                    mask_punctuation = args.mask_punctuation)
    colbert = colbert.to(DEVICE)
  
    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)
  
    colbert.eval()

    return colbert, checkpoint

def create_directory(path):
    if os.path.exists(path):
        print('\n')
        print_message("#> Note: Output directory", path, 'already exists\n\n')
    else:
        print('\n')
        print_message("#> Creating directory", path, '\n\n')
        os.makedirs(path)

def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return

def flatten(L):
    return [x for y in L for x in y]


