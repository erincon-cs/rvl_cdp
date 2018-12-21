import torch
import numpy as np
from torchvision.models import densenet121

import torch.nn as nn
import torch.distributions as tdist


class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()

        self.name = name

    def set_embeddings(self, vectors, freeze=True):
        self.embedding.weight = nn.Parameter(vectors)
        self.embedding.weight.requires_grad = not freeze

    def predict(self, x):
        preds = self.forward(x).cpu()
        preds = self.softmax(preds).numpy()

        preds = np.argmax(preds, axis=1)

        return preds

    def save(self, path):
        torch.save(self.state_dict(), path)


class Convolution2DReparameterization(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Convolution2DReparameterization, self).__init__()

        self.loc = nn.Conv2d(*args, **kwargs)
        self.scale = nn.Conv2d(*args, **kwargs)

        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def forward(self, *input):
        loc = self.loc(input)
        scale = self.loc(input)

        epsilon = self.normal.sample(loc.Size())

        return (loc + scale) * epsilon


class DenseNet121(BaseModel):
    def __init__(self, nb_classes=16, image_shape=(256, 256), pretrained=True):
        super(DenseNet121, self).__init__("DenseNet121")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained

        self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        self.features = nn.Sequential(*list(densenet121(pretrained=pretrained).children())[:-1])
        self.classifier = nn.Linear(8, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_mapping(x)
        x = self.features(x)
        x = self.classifier(x[:, -1, :])

        return x

    def predict(self, x):
        preds = self.forward(x).cpu()
        preds = self.softmax(preds).numpy()

        preds = np.argmax(preds, axis=1)

        return preds
