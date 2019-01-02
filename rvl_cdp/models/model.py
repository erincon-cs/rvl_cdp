import torch
import torch.nn.functional as F

import numpy as np
from torchvision.models import densenet121

import torch.nn as nn
import torch.distributions as tdist


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


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

    def _freeze_layers(self, model):
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        return model

    def save(self, path):
        torch.save(self.state_dict(), path)


class Convolution2DReparameterization(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Convolution2DReparameterization, self).__init__()

        self.loc = nn.Conv2d(*args, **kwargs)
        self.scale = nn.Conv2d(*args, **kwargs)

        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def forward(self, x):
        loc = self.loc(x)
        scale = self.scale(x)

        epsilon = self.normal.sample(loc.size()).squeeze()
        kl = torch.distributions.kl.kl_divergence(tdist.Normal(loc, scale), self.normal)

        return (loc + scale) * epsilon, kl


class LinearReparameterzation(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearReparameterzation, self).__init__()

        self.mean = nn.Linear(*args, **kwargs)
        self.var = nn.Linear(*args, **kwargs)

        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def forward(self, x):
        loc = self.mean(x)
        scale = self.var(x)

        epsilon = self.normal.sample(loc.size()).squeeze()

        if torch.cuda.is_available():
            loc = loc.cuda()
            scale = scale.cuda()
            epsilon = epsilon.cuda()
            
        kl = torch.distributions.kl.kl_divergence(tdist.Normal(loc, scale), self.normal)

        return (loc + scale) * epsilon, kl


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

        return x, kls


class PretrainedBCNN(BaseModel):
    def __init__(self, nb_classes=16, image_shape=(256, 256), pretrained=True):
        super(PretrainedBCNN, self).__init__("PretrainedBCNN")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained

        self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        net = densenet121(pretrained=pretrained)
        net = self._freeze_layers(net)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = LinearReparameterzation(net.classifier.in_features, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_mapping(x)
        x = self.features(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        x = self.classifier(x)

        return x


class DenseNet121(BaseModel):
    def __init__(self, nb_classes=16, image_shape=(256, 256), pretrained=True,
                 feature_extraction_only=False):
        super(DenseNet121, self).__init__("DenseNet121")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained

        self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        net = densenet121(pretrained=pretrained)

        if feature_extraction_only:
            net = self._freeze_layers(net)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Linear(net.classifier.in_features, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_mapping(x)
        x = self.features(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        x = self.classifier(x)

        return x
