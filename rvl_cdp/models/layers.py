import torch

from torch import nn as nn, distributions as tdist
from torch.nn import Parameter, functional as F


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

        loc, scale = torch.tensor([0.0]), torch.tensor([2.5])

        if torch.cuda.is_available():
            loc = loc.cuda()
            scale = scale.cuda()

        self.weight_normal = tdist.Normal(loc, scale)
        self.bias_normal = tdist.Normal(loc, scale)

        self.kl_loss = nn.KLDivLoss()
        self.kl_loss_target_weights = tdist.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        self.kl_loss_target_bias = tdist.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

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
        # weight_normal = tdist.Normal(loc_weight, scale_weight)
        # weight = weight_normal.sample(self.loc_weight.size())

        # bias_normal = tdist.Normal(loc_bias, scale_bias)
        # bias = bias_normal.sample(self.loc_bias.size())
        kl_weight = self.kl_loss(scale_weight, self.kl_loss_target_weights.sample(scale_weight.size()).squeeze())
        kl_bias = self.kl_loss(scale_bias, self.kl_loss_target_bias.sample(scale_bias.size()).squeeze())
        # kl_weight = torch.distributions.kl.kl_divergence(weight_normal, self.weight_normal)
        # kl_bias = torch.distributions.kl.kl_divergence(bias_normal, self.bias_normal)

        kl = kl_weight + kl_bias

        # kl = kl_weight.sum(dim=-1) + kl_bias.sum(dim=-1)
        # kl /= kl_weight.size()[1]

        loc = loc_weight + scale_weight * epsilon_weight
        bias = loc_bias + scale_bias * epsilon_bias

        return F.linear(x, loc, bias), kl


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
