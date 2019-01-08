import torch
import torch.nn.functional as F

import numpy as np

from torchvision.models import densenet121

import torch.nn as nn

from rvl_cdp.models.layers import LinearReparameterzation, Convolution2DReparameterization


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()

        self.name = name
        self.kl = None

    def set_embeddings(self, vectors, freeze=True):
        self.embedding.weight = nn.Parameter(vectors)
        self.embedding.weight.requires_grad = not freeze

    def predict(self, x):
        preds = self.forward(x).cpu()
        preds = self.softmax(preds).numpy()

        preds = np.argmax(preds, axis=1)

        return preds

    def _freeze_layers(self, model):
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        return model

    def save(self, path):
        torch.save(self.state_dict(), path)


class BayesianCNN(BaseModel):
    def __init__(self, nb_classes=16):
        super(BayesianCNN, self).__init__(name="BayesianCNN")

        self.nb_classes = nb_classes

        self.conv1 = Convolution2DReparameterization(1, 200, kernel_size=(3, 3))
        self.activation = nn.PReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv2 = Convolution2DReparameterization(200, 200, kernel_size=(3, 3))
        self.activation_2 = nn.PReLU()
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv3 = Convolution2DReparameterization(200, 100, kernel_size=(3, 3))
        self.activation_3 = nn.PReLU()
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=(3, 3))

        self.classifier = LinearReparameterzation(6400, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        kls = []

        x, kl = self.conv1(x)
        kls.append(kl)

        x = self.activation(x)
        x = self.max_pooling(x)

        # 2
        x, kl = self.conv2(x)
        kls.append(kl)

        x = self.activation_2(x)
        x = self.max_pooling_2(x)
        # 3
        x, kl = self.conv3(x)
        kls.append(kl)

        x = self.activation_3(x)
        x = self.max_pooling_3(x)

        x = Flatten()(x)

        x, kl = self.classifier(x)
        kls.append(kl)

        self.kl = kls

        return x


class PretrainedBCNN(BaseModel):
    def __init__(self, nb_classes=10, image_shape=(256, 256), pretrained=True,
                 two_dim_map=False):

        super(PretrainedBCNN, self).__init__("PretrainedBCNN")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained
        self.two_dim_map = two_dim_map

        self.mean = None
        self.std = None

        if two_dim_map:
            self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        net = densenet121(pretrained=pretrained)
        net = self._freeze_layers(net)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = LinearReparameterzation(net.classifier.in_features, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward_features(self, x):
        if self.two_dim_map:
            x = self.conv_mapping(x)

        x = self.features(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.mean is None or self.std is None:
            mean, std = x.mean(), x.std()
        else:
            mean, std = self.mean, self.std

        x = (x - mean) / (2 * std)

        x, kl = self.classifier(x)
        self.kl = kl

        return x


class DenseNet121(BaseModel):
    def __init__(self, nb_classes=16, image_shape=(256, 256), pretrained=True,
                 feature_extraction_only=False, two_dim_map=False):
        super(DenseNet121, self).__init__("DenseNet121")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained
        self.two_dim_map = two_dim_map

        if self.two_dim_map:
            self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))

        net = densenet121(pretrained=pretrained)

        if feature_extraction_only:
            net = self._freeze_layers(net)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Linear(net.classifier.in_features, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.two_dim_map:
            x = self.conv_mapping(x)
        x = self.features(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        x = self.classifier(x)

        return x
