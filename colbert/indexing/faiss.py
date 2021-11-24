import json
import math
import os
import queue
import threading
from collections import OrderedDict

import torch

from colbert.indexing.faiss_index import FaissIndex
from colbert.indexing.index_manager import load_index_part
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.utils.utils import grouper
from config.config import logger


def get_faiss_index_name(args, offset=None, endpos=None):
    partitions_info = "" if args.partitions is None else f".{args.partitions}"
    range_info = "" if offset is None else f".{offset}-{endpos}"

    return f"ivfpq{partitions_info}{range_info}.faiss"


def load_sample(samples_paths, sample_fraction=None):
    sample = []

    for filename in samples_paths:
        logger.info(f"#> Loading {filename} ...")
        part = load_index_part(filename)
        if sample_fraction:
            part = part[
                torch.randint(
                    0, high=part.size(0), size=(int(part.size(0) * sample_fraction),)
                )
            ]
        sample.append(part)

    sample = torch.cat(sample).float().numpy()

    logger.info("#> Sample has shape {sample.shape}")

    return sample


def prepare_faiss_index(slice_samples_paths, partitions, nprobe, sample_fraction=None):
    training_sample = load_sample(slice_samples_paths, sample_fraction=sample_fraction)

    dim = training_sample.shape[-1]
    index = FaissIndex(dim, partitions, nprobe)

    logger.info("#> Training with the vectors...")

    index.train(training_sample)

    logger.info("Done training!\n")

    return index


SPAN = 3


def index_faiss(args):
    logger.info("#> Starting..")

    parts, parts_paths, samples_paths = get_parts(args.index_path)

    if args.sample is not None:
        assert args.sample, args.sample
        logger.info(
            f"#> Training with {round(args.sample * 100.0, 1)}% of *all* embeddings."
        )
        samples_paths = parts_paths

    num_parts_per_slice = math.ceil(len(parts) / args.slices)

    if not os.path.exists(os.path.join(args.index_path, "num_embeddings")):
        os.makedirs(os.path.join(args.index_path, "num_embeddings"))

    fp = os.path.join(args.index_path, "num_embeddings", "num_embeddings.json")

    for slice_idx, part_offset in enumerate(range(0, len(parts), num_parts_per_slice)):
        part_endpos = min(part_offset + num_parts_per_slice, len(parts))

        # Load previously saved num_embeddings_dict
        if os.path.exists(fp):
            with open(fp) as file:
                num_embeddings_dict = json.load(file)
        else:
            num_embeddings_dict = OrderedDict()

        num_embeddings = sum(
            load_doclens(args.index_path, index=part_offset, step=num_parts_per_slice)
        )
        num_embeddings_dict[part_offset] = num_embeddings

        # Save num_embeddings_dict
        with open(fp, "w") as f:
            json.dump(num_embeddings_dict, f)

        # Calculate partitions for this part
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        logger.info(
            f"Part {part_offset}-{part_endpos}: ",
            "default computation chooses",
            args.partitions,
            f"partitions (for {num_embeddings} embeddings)",
        )

        slice_parts_paths = parts_paths[part_offset:part_endpos]
        slice_samples_paths = samples_paths[part_offset:part_endpos]

        if args.slices == 1:
            faiss_index_name = get_faiss_index_name(args)
        else:
            faiss_index_name = get_faiss_index_name(
                args, offset=part_offset, endpos=part_endpos
            )

        output_path = os.path.join(args.index_path, faiss_index_name)
        logger.info(
            f"#> Processing slice #{slice_idx+1} of {args.slices} (range {part_offset}..{part_endpos})."
        )
        logger.info(f"#> Will write to {output_path}.")

        assert not os.path.exists(output_path), output_path

        # Train faiss index using args.sample of the data
        index = prepare_faiss_index(
            slice_samples_paths, args.partitions, args.nprobe, args.sample
        )

        loaded_parts = queue.Queue(maxsize=1)

        def _loader_thread(thread_parts_paths):
            for filenames in grouper(thread_parts_paths, SPAN, fillvalue=None):
                sub_collection = [
                    load_index_part(filename)
                    for filename in filenames
                    if filename is not None
                ]
                sub_collection = torch.cat(sub_collection)
                sub_collection = sub_collection.float().numpy()
                loaded_parts.put(sub_collection)

        thread = threading.Thread(target=_loader_thread, args=(slice_parts_paths,))
        thread.start()

        logger.info("#> Indexing the vectors...")

        for filenames in grouper(slice_parts_paths, SPAN, fillvalue=None):
            logger.info(f"#> Loading {filenames} (from queue)...")
            sub_collection = loaded_parts.get()

            logger.info(
                f"#> Processing a sub_collection with shape {sub_collection.shape}"
            )
            index.add(sub_collection)

        logger.info("Done indexing!")

        index.save(output_path)

        logger.info(f"Done! All complete (for slice #{slice_idx+1} of {args.slices})!")

        thread.join()
