import torch


def normal_kl(inputs_mean, inputs_std, target_mean, target_std):
    return torch.log(target_std / inputs_std) + target_std ** 2 + (inputs_mean - target_mean) ** 2 \
           / (2 * target_std ** 2) - 1 / 2
