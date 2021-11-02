import torch

from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(
        scores[:, 1].mean().item(), 2
    )
    print(
        "#>>>   ",
        positive_avg,
        negative_avg,
        "\t\t|\t\t",
        round(positive_avg - negative_avg, 3),
    )


def save_checkpoint_pretrain(path, epoch_idx, b_idx, train_loss, model, optimizer):
    print(f"#> Saving a checkpoint to {path}")

    checkpoint = {}
    checkpoint["epoch"] = epoch_idx
    checkpoint["batch"] = b_idx
    checkpoint["train_loss"] = train_loss
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def manage_checkpoints_pretrain(epoch_idx, batch_idx, train_loss, colbert, optimizer):
    path = "colbert/checkpoints/med_marco/"

    if batch_idx in SAVED_CHECKPOINTS:
        name = path + f"biobert-{epoch_idx}-{batch_idx}.pt"
        save_checkpoint_pretrain(
            name, epoch_idx, batch_idx, train_loss, colbert, optimizer
        )


def save_checkpoint(
    path, epoch_idx, b_idx, train_loss, model, optimizer, position_nf, position_bioS
):
    print(f"#> Saving a checkpoint to {path}")

    checkpoint = {}
    checkpoint["epoch"] = epoch_idx
    checkpoint["batch"] = b_idx
    checkpoint["train_loss"] = train_loss
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["position_nf"] = position_nf
    checkpoint["position_bioS"] = position_bioS

    torch.save(checkpoint, path)


def manage_checkpoints(
    epoch_idx, batch_idx, train_loss, colbert, optimizer, position_nf, position_bioS
):
    path = "colbert/checkpoints/bioasq_scifact/"

    if batch_idx in SAVED_CHECKPOINTS:
        name = path + f"biobert-{epoch_idx}-{batch_idx}.pt"
        save_checkpoint(
            name,
            epoch_idx,
            batch_idx,
            train_loss,
            colbert,
            optimizer,
            position_nf,
            position_bioS,
        )
