import os
import rvl_cdp.data.util as data_utils

from rvl_cdp.data.dataset import RVLCDIPDataset, CIFAR10, Food101, RVLCDIPInvoiceDataset

from sklearn.model_selection import train_test_split

_datasets = {
    "rvlcdip": RVLCDIPDataset,
    "cifar10": CIFAR10,
    "food101": Food101,
    "invoices": RVLCDIPInvoiceDataset
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

    train_dataset = Dataset(images_path=images_path, labels_path=train_labels, *args, **kwargs)
    valid_dataset = Dataset(images_path=images_path, labels_path=valid_labels, *args, **kwargs)
    test_dataset = Dataset(images_path=images_path, labels_path=test_labels, *args, **kwargs)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


def load_rvlcdip_invoices(data_path, *args, **kwargs):
    images_path, labels_path = os.path.join(data_path, "images"), os.path.join(data_path, "labels")
    Dataset = get_dataset("invoices")

    train_labels = os.path.join(labels_path, "train.txt")
    valid_labels = os.path.join(labels_path, "val.txt")
    test_labels = os.path.join(labels_path, "test.txt")

    train_dataset = Dataset(images_path=images_path, labels_path=train_labels, *args, **kwargs)
    valid_dataset = Dataset(images_path=images_path, labels_path=valid_labels, *args, **kwargs)
    test_dataset = Dataset(images_path=images_path, labels_path=test_labels, *args, **kwargs)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


def load_cifar10(data_path, *args, **kwargs):
    train_images_path, test_labels_path = os.path.join(data_path, "train"), os.path.join(data_path, "test")

    Dataset = get_dataset("cifar10")

    label_dict, train_data = data_utils.read_image_folders(train_images_path, "*.png")
    _, test_data = data_utils.read_image_folders(train_images_path, "*.png")
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=2019)

    train_data, valid_data = train_data.reset_index(drop=True), valid_data.reset_index(drop=True)

    train_dataset = Dataset(data=train_data, label_dict=label_dict, *args, **kwargs)
    valid_dataset = Dataset(data=valid_data, label_dict=label_dict, *args, **kwargs)
    test_dataset = Dataset(data=test_data, label_dict=label_dict, *args, **kwargs)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


def load_food101(data_path, *args, **kwargs):
    images_path = os.path.join(data_path, "images")
    train_labels_path, test_labels_path = os.path.join(data_path, "train.txt"), os.path.join(data_path, "test.txt")
    Dataset = get_dataset("food101")

    label_dict, train_data = Food101.load_images_labels(images_path, test_labels_path)
    _, test_data = Food101.load_images_labels(images_path, test_labels_path)

    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=2019)
    train_data, valid_data = train_data.reset_index(drop=True), valid_data.reset_index(drop=True)

    train_dataset = Dataset(data=train_data, label_dict=label_dict, *args, **kwargs)
    valid_dataset = Dataset(data=valid_data, label_dict=label_dict, *args, **kwargs)
    test_dataset = Dataset(data=test_data, label_dict=label_dict, *args, **kwargs)

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset}


_dataset_loaders = {
    "rvlcdip": load_rvlcdip,
    "invoices": load_rvlcdip_invoices,
    "cifar10": load_cifar10,
    "food101": load_food101
}


def get_dataset_loader(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_loaders:
        raise ValueError("Dataset loader for {} is not defined!".format(dataset_name))

    return _dataset_loaders[dataset_name]


_dataset_paths = {
    "rvlcdip": "data/rvl_cdip",
    "invoices": "data/rvl_cdip",
    "cifar10": "data/cifar10",
    "food101": "data/food-101"
}


def get_dataset_path(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_loaders:
        raise ValueError("Dataset path for {} is not defined!".format(dataset_name))

    return _dataset_paths[dataset_name]


def load_datasets(dataset_name, data_path=None, *args, **kwargs):
    if data_path is not None:
        data_utils.check_exist_path(data_path)

    if data_path is None:
        data_path = get_dataset_path(dataset_name)

    dataset_loader = get_dataset_loader(dataset_name)

    return dataset_loader(data_path, *args, **kwargs)
