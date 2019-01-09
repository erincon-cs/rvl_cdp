import os
import rvl_cdp.data.util as data_utils

from rvl_cdp.data.dataset import RVLCDIPDataset, CIFAR10
from torchvision.transforms import Compose
from rvl_cdp.data.dataset import Resize, ToTensor, \
    Normalization, RandomImageCrop

from sklearn.model_selection import train_test_split

_datasets = {
    "rvlcdip": RVLCDIPDataset,
    "cifar10": CIFAR10
}


def get_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _datasets:
        raise ValueError("Dataset {} not defined!".format(dataset_name))

    return _datasets[dataset_name]


def load_rvlcdip(data_path, *args, **kwargs):
    images_path, labels_path = os.path.join(data_path, "images"), os.path.join(data_path, "labels")
    Dataset = get_dataset("rvlcdip")

    train_labels = os.path.join(labels_path, "train.txt")
    valid_labels = os.path.join(labels_path, "val.txt")
    test_labels = os.path.join(labels_path, "test.txt")

    train_dataset = Dataset(images_path=images_path, labels_path=train_labels)
    valid_dataset = Dataset(images_path=images_path, labels_path=valid_labels)
    test_dataset = Dataset(images_path=images_path, labels_path=test_labels)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


def load_cifar10(data_path, *args, **kwargs):
    train_images_path, test_labels_path = os.path.join(data_path, "train"), os.path.join(data_path, "test")

    Dataset = get_dataset("cifar10")

    label_dict, train_data = data_utils.read_image_folders(train_images_path, "*.png")
    _, test_data = data_utils.read_image_folders(train_images_path, "*.png")
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=2019)

    train_data, valid_data = train_data.reset_index(drop=True), valid_data.reset_index(drop=True)

    train_dataset = Dataset(data=train_data, label_dict=label_dict)
    valid_dataset = Dataset(data=valid_data, label_dict=label_dict)
    test_dataset = Dataset(data=test_data, label_dict=label_dict)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


_dataset_loaders = {
    "rvlcdip": load_rvlcdip,
    "cifar10": load_cifar10
}


def get_dataset_loader(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_loaders:
        raise ValueError("Dataset loader for {} is not defined!".format(dataset_name))

    return _dataset_loaders[dataset_name]

_dataset_paths = {
    "rvlcdip": "data/",
    "cifar10": load_cifar10
}


def get_dataset_loader(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_loaders:
        raise ValueError("Dataset loader for {} is not defined!".format(dataset_name))

    return _dataset_loaders[dataset_name]


def load_datasets(data_path, dataset_name):
    if data_path is not None:
        data_utils.check_exist_path(data_path)

    if data_path is None:
        data_path = get



    dataset_loader = get_dataset_loader(dataset_name)

    return dataset_loader(data_path)
