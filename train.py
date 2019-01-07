import argparse
import os

from rvl_cdp.training import Trainer
from rvl_cdp.models.factory import get_model

from rvl_cdp.data.factory import load_datasets


def summary_path_path(arg):
    arg = arg.lower()

    if arg == "none":
        return None
    else:
        return arg


def boolean(arg):
    if arg in ("y", "1", ):
        return True
    else:
        return False

_dataset_kwargs = {
    "rvlcdip": {"two_dim_map": True},
    "cifar10": {}
}


def get_model_kwargs(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_kwargs:
        raise ValueError("Dataset {} not defined!".format(dataset_name))

    return _dataset_kwargs[dataset_name]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="dn121")
    arg_parser.add_argument("--dataset", default='rvlcdip', type=str,
                            help="Name of the dataset.\nThe default dataset is rvlcdip"
                                 "which is just the raw RVL-CDIP data")
    arg_parser.add_argument("--data", type=str, help="Directory where the data is stored")
    arg_parser.add_argument("--num-workers", default=3, type=int,
                            help="The number of workers to use in the DataLoader")
    arg_parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    arg_parser.add_argument("--mb-size", default=64, type=int, help='Minibatch size')
    arg_parser.add_argument("--stats", default="")
    arg_parser.add_argument("--summary-path", default="none", type=summary_path_path)
    args = arg_parser.parse_args()

    datasets = load_datasets(args.data, args.dataset)

    Model = get_model(args.model)
    model_kwwargs = get_model_kwargs(args.dataset)
    model = Model(**model_kwwargs)

    trainer = Trainer(model, summary_path=args.summary_path, **datasets)
    trainer.fit(learning_rate=args.lr, minibatch_size=args.mb_size)
    trainer.evaluate(datasets["test_dataset"])


if __name__ == '__main__':
    main()
