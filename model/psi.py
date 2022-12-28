import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_
from model.blocks import FeedForward
from torch.nn.utils.parametrizations import orthogonal


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, causal=False, heads=heads, dim_head=head_dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        y = self.attn(self.norm1(x), mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class MyTransformer(nn.Module):
    def __init__(self, num_blocks, k, d_backbone, mlp_dim, num_heads=None, head_dim=64, 
                 normalize_over=[0, 1], momentum=0.99):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = d_backbone * 3
        self.psi_dim = k
        self.normalize_over = normalize_over
        self.momentum = momentum

        if num_heads is None:
            num_heads = self.hidden_dim//head_dim
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(Block(self.hidden_dim, num_heads, head_dim, mlp_dim))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = orthogonal(nn.Linear(self.hidden_dim, self.psi_dim, bias=True))

        # self.register_buffer('eigennorm', torch.zeros(self.psi_dim))
        # self.register_buffer('num_calls', torch.Tensor([0]))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["bias"])}

    def forward(self, x, tau=None):
        hidden = self.norm(self.blocks(x))
        ret_raw = self.head(hidden)
        if tau is not None:
            ret_raw = torch.nn.functional.gumbel_softmax(ret_raw, tau=tau, hard=False)
            # ret_raw = torch.nn.functional.softmax(ret_raw, dim=-1)

        if self.training:
            norm_ = ret_raw.norm(dim=self.normalize_over).clamp(min=1)
            # with torch.no_grad():
            #     if self.num_calls == 0:
            #         self.eigennorm.copy_(norm_.data)
            #     else:
            #         self.eigennorm.mul_(self.momentum).add_(
            #             norm_.data, alpha = 1-self.momentum)
            #     self.num_calls += 1
        else:
            norm_ = 1 #self.eigennorm
        return hidden, ret_raw / norm_