import os

import ujson


def get_parts(directory, index=None, step=None):
    extension = ".pt"

    parts = sorted(
        int(filename[: -1 * len(extension)])
        for filename in os.listdir(directory)
        if filename.endswith(extension)
    )

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [
        os.path.join(directory, f"{filename}{extension}") for filename in parts
    ]
    samples_paths = [
        os.path.join(directory, f"{filename}.sample") for filename in parts
    ]

    if index is not None:
        parts, parts_paths, samples_paths = (
            parts[index : index + step],
            parts_paths[index : index + step],
            samples_paths[index : index + step],
        )

    return parts, parts_paths, samples_paths


def load_doclens(directory, index=None, step=None, flatten=True):
    parts, _, _ = get_parts(directory, index, step)

    doclens_filenames = [
        os.path.join(directory, f"doclens.{filename}.json") for filename in parts
    ]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens
