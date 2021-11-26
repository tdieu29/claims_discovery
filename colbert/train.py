import wandb

from colbert.parameters import train_params
from colbert.training.training import train
from colbert.utils.parser import Arguments


def main():
    parser = Arguments()

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    params_list = train_params()
    args = parser.parse(params_list)

    assert args.bsize % args.accumsteps == 0, (
        (args.bsize, args.accumsteps),
        "The batch size must be divisible by the number of gradient accumulation steps.",
    )
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    with wandb.init(
        project="knowledge-discovery", job_type="train", config=args
    ) as run:
        config = wandb.config

        train(config)


if __name__ == "__main__":
    main()
