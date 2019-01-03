import torch
import torch.nn.functional as F

import numpy as np

from torch.nn.parameter import Parameter
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
    def __init__(self, in_features, out_features, bias=True):
        super(LinearReparameterzation, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        loc_weight = torch.Tensor(out_features, in_features)
        self.init_weight(loc_weight)
        scale_weight = torch.Tensor(out_features, in_features)
        self.init_weight(scale_weight)

        self.loc_weight = Parameter(loc_weight)
        self.scale_weight = Parameter(scale_weight)

        loc, scale = torch.tensor([0.0]), torch.tensor([1.0])

        if torch.cuda.is_available():
            loc = loc.cuda()
            scale = scale.cuda()

        self.weight_normal = tdist.Normal(loc, scale)
        self.bias_normal = tdist.Normal(loc, scale)

        if bias:
            loc_bias = torch.Tensor(out_features)
            self.init_bias(loc_bias)

            scale_bias = torch.Tensor(out_features)
            self.init_bias(scale_bias)

            self.loc_bias = Parameter(loc_bias)
            self.scale_bias = Parameter(scale_bias)
        else:
            self.register_parameter('bias', None)

    def init_weight(self, m):
        torch.nn.init.xavier_uniform(m)

    def init_bias(self, b):
        torch.nn.init.constant(b, 1.0)

    def forward(self, x):

        epsilon_weight = self.weight_normal.sample(self.loc_weight.size()).squeeze()
        epsilon_bias = self.bias_normal.sample(self.loc_bias.size()).squeeze()

        loc_weight, scale_weight = self.loc_weight, self.scale_weight
        loc_bias, scale_bias = self.loc_bias, self.scale_bias

        if torch.cuda.is_available():
            loc_weight, scale_weight = loc_weight.cuda(), scale_weight.cuda()
            loc_bias, scale_bias = loc_bias.cuda(), scale_bias.cuda()
            epsilon_weight = epsilon_weight.cuda()
            epsilon_bias = epsilon_bias.cuda()

        # log transform
        scale_weight, scale_bias = F.softplus(scale_weight), F.softplus(scale_bias)
        weight_normal = tdist.Normal(loc_weight, scale_weight)
        # weight = weight_normal.sample(self.loc_weight.size())

        bias_normal = tdist.Normal(loc_bias, scale_bias)
        # bias = bias_normal.sample(self.loc_bias.size())

        kl_weight = torch.distributions.kl.kl_divergence(weight_normal, self.weight_normal)
        kl_bias = torch.distributions.kl.kl_divergence(bias_normal, self.bias_normal)

        kl = kl_weight.sum(dim=-1) + kl_bias.sum(dim=-1)
        kl /= kl_weight.size()[1]

        loc = loc_weight + scale_weight * epsilon_weight
        bias = loc_bias + scale_bias * epsilon_bias

        return F.linear(x, loc, bias), kl


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
    def __init__(self, nb_classes=10, image_shape=(256, 256), pretrained=True,
                 two_dim_map=False):

        super(PretrainedBCNN, self).__init__("PretrainedBCNN")

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained
        self.two_dim_map = two_dim_map

        if two_dim_map:
            self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        net = densenet121(pretrained=pretrained)
        net = self._freeze_layers(net)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = LinearReparameterzation(net.classifier.in_features, self.nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.two_dim_map:
            x = self.conv_mapping(x)

        x = self.features(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        x, kl = self.classifier(x)

        return x, kl


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
