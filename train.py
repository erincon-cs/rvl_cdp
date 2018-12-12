import argparse
import os

from rvl_cdp.training import Trainer
from rvl_cdp.models.factory import get_model
from rvl_cdp.data.factory import get_dataset


def check_exist_path(path):
    if not os.path.exists(path):
        raise ValueError("Path {} does not exist".format(path))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="dn121")
    arg_parser.add_argument("--dataset", default='rvlcdip', type=str,
                            help="Name of the dataset.\nThe default dataset is rvlcdip"
                                 "which is just the raw RVL-CDIP data")
    arg_parser.add_argument("--images", default='data/images',
                            type=str, help="Directory path to the images")
    arg_parser.add_argument("--labels", default="data/labels",
                            help="Directory path to the labels")
    arg_parser.add_argument("--num-workers", default=3, type=int,
                            help="The number of workers to use in the DataLoader")
    args = arg_parser.parse_args()

    Dataset = get_dataset("rvlcdip")

    check_exist_path(args.labels)
    check_exist_path(args.images)

    train_labels = os.path.join(args.labels, "train.txt")
    valid_labels = os.path.join(args.labels, "val.txt")
    test_labels = os.path.join(args.labels, "test.txt")

    train_dataset = Dataset(images_path=args.images, labels_path=train_labels)
    valid_dataset = Dataset(images_path=args.images, labels_path=valid_labels)
    test_dataset = Dataset(images_path=args.images, labels_path=test_labels)

    Model = get_model(args.model)
    model = Model()
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset)
    trainer.fit()
    trainer.evaluate(test_dataset)


if __name__ == '__main__':
    main()
