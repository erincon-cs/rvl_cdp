from rvl_cdp.data.dataset import RVLCDIPDataset

_datasets = {
    "rvlcdip": RVLCDIPDataset
}


def get_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _datasets:
        raise ValueError("Dataset {} not defined!".format(dataset_name))

    return _datasets[dataset_name]
