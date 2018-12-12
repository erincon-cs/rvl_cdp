import torch
from torchvision.models import densenet121

import torch.nn as nn


class DenseNet121(nn.Module):
    def __init__(self, nb_classes=16, image_shape=(256, 256), pretrained=True):
        super(DenseNet121, self).__init__()

        self.nb_classes = nb_classes
        self.imgae_shape = image_shape
        self.pretrained = pretrained

        self.conv_mapping = nn.Conv2d(1, 3, kernel_size=(1, 1))
        self.features = nn.Sequential(*list(densenet121(pretrained=pretrained).children())[:-1])
        self.classifier = nn.Linear(8, self.nb_classes)

    def __call__(self, x, *args, **kwargs):
        x = self.conv_mapping(x)
        x = self.features(x)
        x = self.classifier(x[:, -1, :])

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
