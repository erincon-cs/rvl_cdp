from rvl_cdp.models.model import DenseNet121, BayesianCNN

_models = {
    "dn121": DenseNet121,
    "bcnn": BayesianCNN
}


def get_model(model_name):
    model_name = model_name.lower()

    if model_name not in _models:
        raise ValueError("Model name {} not defined!".format(model_name))

    return _models[model_name]
