import argparse
import os

from rvl_cdp.training import Trainer
from rvl_cdp.models.factory import get_model
from rvl_cdp.data.factory import get_dataset


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="dn121")
    arg_parser.add_argument("--dataset", default='rvlcdip')
    arg_parser.add_argument("--images")
    arg_parser.add_argument("--labels")
    args = arg_parser.parse_args()

    Dataset = get_dataset("rvlcdip")

    if not os.path.exists(args.images):
        raise ValueError("Path {} does not exist".format(args.images))

    if not os.path.exists(args.labels):
        raise ValueError("Path {} does not exist".format(args.images))

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