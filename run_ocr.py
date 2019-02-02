import os
import argparse
import pandas as pd

from rvl_cdp.data.factory import load_datasets
from rvl_cdp.models.ocr import Tesseract

from rvl_cdp.arg_utils import none_arg_path

from rvl_cdp.data.dataset import NoTransforms
from tqdm import tqdm


def run(datasets, output_path, ocr_model):
    for dataset_type, dataset in datasets.items():
        print("Reading text from {}".format(dataset_type))
        df = pd.DataFrame({"text": ["" for _ in range(len(dataset))]})

        for sample in tqdm(dataset):
            image, label, path, idx = sample["image"], sample["label"], sample["path"], sample['idx']
            text = ocr_model.get_text(image)

            df["text"] = text

        df.to_csv(os.path.join(output_path, "{}.csv".format(dataset_type)))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", default="rvlcdip")
    arg_parser.add_argument("--output-path", default="data/rvl_cdip")
    arg_parser.add_argument("--data-path", default="none", type=none_arg_path)
    arg_parser.add_argument("--tesseract-path", default="none", type=none_arg_path)
    args = arg_parser.parse_args()

    datasets = load_datasets(args.dataset, data_path=args.data_path, transforms=NoTransforms())
    ocr_model = Tesseract(path=args.tesseract_path)

    run(datasets, args.output_path, ocr_model)


if __name__ == '__main__':
    main()
