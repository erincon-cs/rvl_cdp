def none_arg_path(arg):
    arg = arg.lower()

    if arg == "none":
        return None
    else:
        return arg


def boolean(arg):
    if arg in ("y", "yes", "1", "true", "t"):
        return True
    else:
        return False


_dataset_kwargs = {
    "rvlcdip": {"two_dim_map": True},
    "cifar10": {"nb_classes": 10},
    "food101": {"nb_classes": 101}
}


def get_model_kwargs(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _dataset_kwargs:
        raise ValueError("Dataset {} not defined!".format(dataset_name))

    model_kwargs = _dataset_kwargs[dataset_name]

    return model_kwargs