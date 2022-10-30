import math
import torch
import torch.nn as nn

from model.utils import init_weights
from functools import partial
import numpy as np


class MLP(nn.Module):
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0], momentum=0.99):
        super().__init__()
        self.normalize_over = normalize_over
        self.momentum = momentum

        if norm_type.lower() == 'bn':
            norm_fn = nn.BatchNorm1d
        elif norm_type.lower() == 'ln':
            norm_fn = partial(nn.LayerNorm, eps=1e-6)
        else:
            norm_fn = nn.Identity
        
        if act_type.lower() == 'relu':
            act_fn = nn.ReLU
        elif act_type.lower() == 'gelu':
            act_fn = nn.GELU
        elif act_type.lower() == 'sigmoid':
            act_fn = nn.Sigmoid
        else:
            act_fn = nn.Identity
        
        sizes = [d_backbone,] + [num_features,]*(num_layers - 1) + [k]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False if norm_type is not None else True))
            layers.append(norm_fn(sizes[i+1]))
            layers.append(act_fn())
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
        self.fn = nn.Sequential(*layers)

        self.register_buffer('eigennorm', torch.zeros(k))
        self.register_buffer('num_calls', torch.Tensor([0]))

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["bias"])}

    def forward(self, x):
        ret_raw = self.fn(x)

        if self.training:
            norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
                np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm.copy_(norm_.data)
                else:
                    self.eigennorm.mul_(self.momentum).add_(
                        norm_.data, alpha = 1-self.momentum)
                self.num_calls += 1
        else:
            norm_ = self.eigennorm
        return ret_raw / norm_
