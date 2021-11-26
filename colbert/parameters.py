import torch

# Device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Saved checkpoints
A = 100000
SAVED_CHECKPOINTS = [A, A * 2, A * 3, A * 4, A * 5, A * 6, A * 7, A * 8, A * 9, A * 10]


# Parameters for train.py
def train_params():
    return [
        "amp",
        # Uncomment below when necessary
        # "resume",
        # "resume_optimizer",
        # "checkpoint", ""
        # "triples", "",
        # "queries", "",
        # "collection", ""
    ]


# Parameters for index.py
def index_params():
    return [
        "amp",
        "checkpoint",
        "colbert/model_checkpoint/biobert-MM-2970159.pt",
        "index_name",
        "BioSciNF.L2.8x2970159",
    ]


# Parameters for index_faiss.py
def index_faiss_params():
    return ["index_name", "BioSciNF.L2.8x2970159", "slices", 39]


# Parameters for retrieve.py
def retrieve_params():
    return [
        "amp",
        "checkpoint",
        "colbert/model_checkpoint/biobert-MM-2970159.pt",
        "index_name",
        "BioSciNF.L2.8x2970159",
        "slices",
        39,
        "num_faiss_indexes",
        39,
        # "queries", "", # Uncomment if necessary
    ]
