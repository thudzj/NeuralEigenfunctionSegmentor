import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from model.utils import init_weights
from functools import partial
import numpy as np


class MLP(nn.Module):
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0], momentum=0.99):
        super().__init__()
        self.num_layers = num_layers
        self.out_features = k
        if self.num_layers == 0:
            self.out_features = d_backbone
            return

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
        if self.num_layers == 0:
            return x

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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, norm_fn, act_fn, planes, drop_path=0):
        super(BasicBlock, self).__init__()
        self.linear1 = nn.Linear(planes, planes, bias=False)
        self.bn1 = norm_fn(planes)
        self.act1 = act_fn()
        self.linear2 = nn.Linear(planes, planes, bias=False)
        self.bn2 = norm_fn(planes)
        self.act2 = act_fn()
        self.shortcut = nn.Sequential()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0], momentum=0.99):
        super().__init__()
        self.num_layers = num_layers
        self.out_features = k
        if self.num_layers == 0:
            self.out_features = d_backbone
            return

        self.normalize_over = normalize_over
        self.momentum = momentum

        if norm_type.lower() == 'bn':
            norm_fn = nn.BatchNorm1d
        elif norm_type.lower() == 'ln':
            norm_fn = partial(nn.LayerNorm, eps=1e-6)
        else:
            norm_fn = nn.Identity
        
        if act_type.lower() == 'relu':
            act_fn = partial(nn.ReLU, inplace=True)
        elif act_type.lower() == 'gelu':
            act_fn = nn.GELU
        elif act_type.lower() == 'sigmoid':
            act_fn = nn.Sigmoid
        else:
            act_fn = nn.Identity

        self.linear1 = nn.Linear(d_backbone, num_features,  bias=False)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.res_layers = self._make_layer(BasicBlock, norm_fn, act_fn, num_features, num_layers // 2 - 1)
        self.head = nn.Linear(num_features, self.out_features)

        self.register_buffer('eigennorm', torch.zeros(k))
        self.register_buffer('num_calls', torch.Tensor([0]))

    def _make_layer(self, block, norm_fn, act_fn,  planes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(norm_fn, act_fn, planes))
        return nn.Sequential(*layers)

    def no_weight_decay(self):
        return {}

    def forward(self, x):

        if self.num_layers == 0:
            return x

        out = F.relu(self.bn1(self.linear1(x)))
        out = self.res_layers(out)
        ret_raw = self.head(out)
        
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