import torch
import pandas as pd
import os

from collections import OrderedDict
import rvl_cdp.data.util as data_utils

from skimage import io

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from rvl_cdp.data.transforms import Resize, Normalization, ToTensor, PermuteTensor


def one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).long()

    return y[labels]


def read_textfile(image_paths, path):
    examples = []

    with open(path, "r") as file:
        for line in file.readlines():
            image_path, label = line.split()
            image_path = os.path.join(image_paths, image_path)

            examples.append({"path": image_path, "label": int(label)})



    return pd.DataFrame(examples, columns=["path", "label"])


class BaseDataset(Dataset):
    def __init__(self, nb_classes, images_path=None, labels_path=None, data=None, label_dict=None,
                 transforms=None):
        super(BaseDataset, self).__init__()

        if transforms is None:
            transforms = Compose([
                Normalization(),
                Resize(),
                ToTensor()
            ])

        self.transforms = transforms
        self.nb_classes = nb_classes
        self.images_path = images_path
        self.labels_path = labels_path

        self.label_dict = label_dict if label_dict is not None else OrderedDict()
        self.data = data if data is not None else self.read_data()

    def __getitem__(self, idx):
        print(idx)
        image_path, label = self.data.path.iloc[idx], self.data.label.iloc[idx]

        image = io.imread(image_path)

        if self.transforms:
            sample = self.transforms({"image": image, "label": label})
            image, label = sample["image"], sample['label']

        return {"image": image, "label": one_hot(label, self.nb_classes)}

    def __len__(self):
        return len(self.data)

    def read_data(self):
        pass


class RVLCDIPDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        if "transforms" not in kwargs:
            transforms = Compose([
                Resize(),
                Normalization(),
                ToTensor(unsqueeze=True),
                PermuteTensor((2, 0, 1))
            ])

            kwargs["transforms"] = transforms


        super(RVLCDIPDataset, self).__init__(*args, nb_classes=16, **kwargs)


        if os.path.exists("data/rvl_cdip/errors.txt"):
            with open("data/rvl_cdip/errors.txt") as error_files:
                error_paths = error_files.readlines()

                self.data = self.data[~self.data.path.isin(error_paths)]




    def read_data(self):
        return read_textfile(self.images_path, self.labels_path)


class CIFAR10(BaseDataset):
    def __init__(self, *args, **kwargs):
        if "transforms" not in kwargs:
            transforms = Compose([
                Resize(),
                Normalization(),
                ToTensor(),
                PermuteTensor((2, 0, 1))
            ])
            kwargs["transforms"] = transforms

        super(CIFAR10, self).__init__(*args, nb_classes=10, **kwargs)

    def read_data(self):
        label_dict, data = data_utils.read_image_folders(self.images_path, "*.png")

        self.label_dict = label_dict

        return data

