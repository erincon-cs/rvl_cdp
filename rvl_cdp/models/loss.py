import torch

from torch import nn


class KLNormal(nn.Module):
    def __init__(self):
        super(KLNormal, self).__init__()

    def forward(self, inputs_mean, inputs_std, target_mean, target_std):
        return torch.log(target_std) - torch.log(inputs_std) + target_std ** 2 + (((inputs_mean - target_mean) ** 2) \
                                                                                  / (2 * target_std ** 2)) - 1 / 2
