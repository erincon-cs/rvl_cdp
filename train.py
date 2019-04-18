import argparse

from rvl_cdp.arg_utils import none_arg_path, boolean, get_model_kwargs
from rvl_cdp.training import Trainer
from rvl_cdp.models.factory import get_model

from rvl_cdp.data.factory import load_datasets


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="dn121")
    arg_parser.add_argument("--dataset", default='rvlcdip', type=str,
                            help="Name of the dataset.\nThe default dataset is rvlcdip"
                                 "which is just the raw RVL-CDIP data")
    arg_parser.add_argument("--data-path", help="Directory where the data is stored", type=none_arg_path)
    arg_parser.add_argument("--num-workers", default=3, type=int,
                            help="The number of workers to use in the DataLoader")
    arg_parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    arg_parser.add_argument("--mb-size", default=64, type=int, help='Minibatch size')
    arg_parser.add_argument("--stats", default="")
    arg_parser.add_argument("--epochs", default=20, type=int)
    arg_parser.add_argument("--summary-path", default="none", type=none_arg_path)
    arg_parser.add_argument("--weight-hists", default="y", type=boolean)
    arg_parser.add_argument("--save-model", default="n", type=boolean)
    arg_parser.add_argument("--save-dir", default="model", type=str)
    arg_parser.add_argument("--finetune", default="n", type=boolean)
    arg_parser.add_argument("--pretrained", default="y", type=boolean)
    arg_parser.add_argument("--output-dir", default="model_output", type=str)
    args = arg_parser.parse_args()

    datasets = load_datasets(args.dataset, data_path=args.data_path)

    Model = get_model(args.model)
    model_kwwargs = get_model_kwargs(args)
    print("-" * 100)
    print("Model parameters:")
    print(model_kwwargs)
    print("-" * 100)
    model = Model(**model_kwwargs)

    trainer = Trainer(
        model,
        summary_path=args.summary_path,
        model_path=args.output_dir,
        weight_histograms=args.weight_hists,
        **datasets
    )

    trainer.fit(learning_rate=args.lr, minibatch_size=args.mb_size, nb_epochs=args.epochs)
    metric_results, avg_loss = trainer.evaluate(datasets["test_dataset"], save=args.save_model)

    print("Model metrics")
    print("Test loss: {0:.2f}".format(avg_loss))

    for metric_name, metric_score in metric_results.items():
        print("Test {} score: ".format(metric_name), "{0:.2f}".format(metric_score))

if __name__ == '__main__':
    main()
