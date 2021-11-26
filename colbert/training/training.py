import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from transformers import AdamW

from colbert.modeling.colbert import ColBERT
from colbert.parameters import DEVICE
from colbert.training.batcher import Batcher
from colbert.training.batcher_pretrain import Batcher_Pretrain
from colbert.training.utils import (
    log_progress,
    manage_checkpoints,
    manage_checkpoints_pretrain,
)
from colbert.utils.amp import MixedPrecisionManager
from config.config import logger


def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    if args.pretrain:
        reader = Batcher_Pretrain(args)
    else:
        reader = Batcher(args)

    colbert = ColBERT(
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )

    if args.checkpoint is not None:
        logger.info(f"#> Starting from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        try:
            logger.info("#> Loading model state dict")
            colbert.load_state_dict(checkpoint["model_state_dict"])
        except:
            logger.warning("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint["model_state_dict"], strict=False)

    colbert = colbert.to(DEVICE)
    colbert.train()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8
    )

    if args.resume_optimizer is True:
        assert args.checkpoint is not None
        logger.info("#> Loading optimizer state dict")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0
    start_batch_idx = 0

    if args.resume:
        if args.pretrain:
            assert args.checkpoint is not None

            start_batch_idx = checkpoint["batch"]
            reader.skip_to_batch(start_batch_idx, args.bsize)

            train_loss = checkpoint["train_loss"]
        else:
            assert args.checkpoint is not None

            start_batch_idx = checkpoint["batch"]
            position_nf = checkpoint["position_nf"]
            position_bioS = checkpoint["position_bioS"]
            switch_to_bioS = checkpoint["switch_to_bioS"]
            count = checkpoint["count"]
            reader.skip_to_batch(
                start_batch_idx,
                args.bsize,
                position_nf,
                position_bioS,
                switch_to_bioS,
                count,
            )

            train_loss = checkpoint["train_loss"]

    for epoch_idx in range(args.num_epochs):
        for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
            this_batch_loss = 0.0

            if args.pretrain:
                batch = BatchSteps
            else:
                batch, position_nf, position_bioS, switch_to_bioS, count = BatchSteps

            for queries, passages in batch:

                # Get the attention mask of passages
                (
                    _,
                    psg_attn_mask,
                ) = passages  # shape = torch.Size([batch size, doc_seq_len])

                with amp.context():
                    scores = (
                        colbert(queries, passages, psg_attn_mask)
                        .view(2, -1)
                        .permute(1, 0)
                    )
                    loss = criterion(scores, labels[: scores.size(0)])
                    loss = (loss / 2) / args.accumsteps

                    log_progress(scores)

                amp.backward(loss)

                train_loss += loss.item()
                this_batch_loss += loss.item()

            if (batch_idx + 1) % args.accumsteps == 0:
                amp.step(colbert, optimizer)

                avg_loss = train_loss / (batch_idx + 1)

                num_examples_seen = (batch_idx - start_batch_idx) * args.bsize
                elapsed = float(time.time() - start_time)

                # Log messages
                logger.info(epoch_idx, batch_idx, avg_loss)
                if not args.pretrain:
                    logger.info(
                        "position_nf: ",
                        position_nf,
                        "\t\t|\t\t",
                        "position_bioS: ",
                        position_bioS,
                    )
                    logger.info(
                        "switch_to_bioS: ",
                        switch_to_bioS,
                        "\t\t|\t\t",
                        "count: ",
                        count,
                    )

            # Log every ten batches
            log = (batch_idx + 1) % 10 == 0

            if log:
                wandb.log(
                    {
                        "avg_loss": avg_loss,
                        "batch_loss": this_batch_loss,
                        "examples_seen": num_examples_seen,
                        "throughput": num_examples_seen / elapsed,
                    }
                )
            if args.pretrain:
                manage_checkpoints_pretrain(
                    epoch_idx, batch_idx + 1, train_loss, colbert, optimizer
                )
            else:
                manage_checkpoints(
                    epoch_idx,
                    batch_idx + 1,
                    train_loss,
                    colbert,
                    optimizer,
                    position_nf,
                    position_bioS,
                    switch_to_bioS,
                    count,
                )
