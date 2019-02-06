import torch
import math

from torch import nn as nn, distributions as tdist
from torch.nn import Parameter, functional as F

from rvl_cdp.models.loss import KLNormal

_transforms = {
    "sofplus": F.softplus,
    "exp": torch.exp
}

_dists = {
    "normal": tdist.Normal,
    "cuachy": tdist.Cauchy
}


def _get_transform(transform_name):
    transform_name = transform_name.lower()

    if transform_name not in _transforms:
        raise ValueError("Transformer function {} not defined!".format(transform_name))

    return _transforms[transform_name]


def _get_dist(dist_name):
    dist_name = dist_name.lower()

    if dist_name not in _dists:
        raise ValueError("Distribution function {} not defined!".format(dist_name))

    return _dists[dist_name]


class LinearReparameterzation(nn.Module):
    def __init__(self, in_features, out_features, bias=True, transformer_name="exp",
                 prior_dist='normal'):
        super(LinearReparameterzation, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.loc_scale = 2.5
        self.scale_bias = 10
        self.loc_bias_weight = 10
        self.dist = _get_dist(prior_dist)

        loc_weight = torch.Tensor(out_features, in_features)
        self.init_weight(loc_weight)
        self.loc_weight = Parameter(loc_weight)
        self.register_parameter("loc_weight", self.loc_weight)

        scale_weight = torch.Tensor(out_features, in_features)
        self.init_weight(scale_weight)
        self.scale_weight = Parameter(scale_weight)
        self.register_parameter("scale_weight", self.scale_weight)

        loc, log_scale = torch.tensor([0.0]), torch.tensor([self.loc_scale])
        scale_bias = torch.tensor([self.loc_bias_weight])

        if torch.cuda.is_available():
            loc = loc.cuda()
            log_scale = log_scale.cuda()
            scale_bias = scale_bias.cuda()

        self.weight_normal = tdist.Normal(loc, log_scale)
        self.bias_normal = tdist.Normal(loc, log_scale)

        self.kl_loss_weights = KLNormal()
        self.kl_loss_bias = KLNormal()

        self.kl_loss_target_weights = tdist.Normal(loc.clone(), log_scale.clone())
        self.kl_loss_target_bias = tdist.Normal(loc.clone(), scale_bias)

        self.scale_transform = _get_transform(transformer_name)

        if bias:
            loc_bias = torch.Tensor(out_features)
            self.init_bias(loc_bias)

            scale_bias = torch.Tensor(out_features)
            self.init_bias(scale_bias)

            self.loc_bias, self.scale_bias = Parameter(loc_bias), Parameter(scale_bias)
            self.register_parameter("loc_bias", self.loc_bias)
            self.register_parameter("scale_bias", self.scale_bias)

    def init_bias(self, m):
        torch.nn.init.constant(m, math.log(math.e, self.scale_bias))

    def init_weight(self, m):
        torch.nn.init.constant(m, 2 * math.log(math.e, self.loc_scale))

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

        scale_weight, scale_bias = self.scale_transform(scale_weight), self.scale_transform(scale_bias)

        if self.training:
            loc = loc_weight + scale_weight * epsilon_weight
            bias = scale_bias + scale_bias * epsilon_bias
        else:

            loc = loc_weight
            bias = loc_bias

        weight_mean, weight_std = torch.Tensor([0.0]), torch.Tensor([2.5])
        bias_mean, bias_std = torch.Tensor([0.0]), torch.Tensor([10])

        if torch.cuda.is_available():
            weight_mean, weight_std = weight_mean.cuda(), weight_std.cuda()
            bias_mean, bias_std = bias_mean.cuda(), bias_std.cuda()

        kl_weight = sum(sum(self.kl_loss_weights(loc_weight, scale_weight, weight_mean, weight_std)))
        kl_bias = sum(self.kl_loss_bias(loc_bias, scale_bias, bias_mean, bias_std))
        kl = kl_weight + kl_bias

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
