import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from model.utils import init_weights
from functools import partial
import numpy as np

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
from timm.models.layers import Mlp

from model.blocks import FeedForward, Attention

from performer_pytorch import SelfAttention as PerformerSelfAttention
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_




class MLP(nn.Module):
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0], momentum=0.99, projector=False, upsample_factor=1):
        super().__init__()
        self.num_layers = num_layers
        self.out_features = k
        if self.num_layers == 0:
            self.out_features = d_backbone
            return

        self.normalize_over = normalize_over
        self.momentum = momentum
        self.num_features = num_features
        self.projector = projector
        self.upsample_factor = upsample_factor

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
        
        sizes = [d_backbone * 3,] + [num_features,]*(num_layers - 1) + [k * (self.upsample_factor ** 2)]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
            acl_l = act_fn()
            norm_l = norm_fn(sizes[i+1])
            if isinstance(norm_l, nn.LayerNorm) and isinstance(acl_l, nn.GELU):
                layers.append(acl_l)
                layers.append(norm_l)
            else:
                layers.append(norm_l)
                layers.append(acl_l)
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
        self.fn = nn.Sequential(*layers)
        # self.shortcut = (nn.Linear(sizes[0], sizes[-1]))

        if self.projector:
            self.fn[-1] = nn.Identity()
            sizes = [num_features, 2048, self.out_features]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
            self.projector_fn = nn.Sequential(*layers)

        self.register_buffer('eigennorm', torch.zeros(k))
        self.register_buffer('num_calls', torch.Tensor([0]))

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["bias"])}

    def forward(self, x):
        x_shape = x.shape
        x = x.flatten(0, 1)
        if self.num_layers == 0:
            return x, x

        hidden = self.fn(x)# + self.shortcut(x)
        if self.projector:
            ret_raw = self.projector_fn(hidden)
        else:
            ret_raw = hidden
            
        ret_raw = rearrange(ret_raw, "(b w h) (c d l) -> b (w c) (h d) l", b=x_shape[0], w=int(math.sqrt(x_shape[1])), c=self.upsample_factor, d=self.upsample_factor).flatten(0, 2)
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
        
        if self.projector:
            return hidden, ret_raw / norm_
        else:
            ret = ret_raw / norm_
            return ret, ret



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
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0], momentum=0.99, projector=False):
        super().__init__()
        self.num_layers = num_layers
        self.out_features = k
        if self.num_layers == 0:
            self.out_features = d_backbone
            return

        self.normalize_over = normalize_over
        self.momentum = momentum
        self.num_features = num_features
        self.projector = projector

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

        if self.projector:
            self.head = nn.Identity()
            sizes = [num_features, 2048, self.out_features]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
            self.projector_fn = nn.Sequential(*layers)

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
            return x, x

        out = F.relu(self.bn1(self.linear1(x)))
        out = self.res_layers(out)
        hidden = self.head(out)
        if self.projector:
            ret_raw = self.projector_fn(hidden)
        else:
            ret_raw = hidden
        
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

        if self.projector:
            return hidden, ret_raw / norm_
        else:
            ret = ret_raw / norm_
            return ret, ret

class MyAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        norm_type='ln',
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.map = nn.Linear(dim, inner_dim*3, bias = False)
        self.norm_k = nn.LayerNorm(dim_head) if norm_type == 'ln' else nn.InstanceNorm1d(dim_head)
        self.norm_v = nn.LayerNorm(dim_head) if norm_type == 'ln' else nn.InstanceNorm1d(dim_head)
        self.to_out = nn.Linear(inner_dim, dim)# if inner_dim != dim else nn.Identity()

    def forward(
        self,
        x,
    ):
        h = self.heads
        q, k, v = self.map(x).chunk(3, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        if isinstance(self.norm_k, nn.LayerNorm):
            k = self.norm_k(k)
            v = self.norm_v(v)
        else:
            k = self.norm_k(k.flatten(0, 1).permute(0, 2, 1)).permute(0, 2, 1).view_as(k)
            v = self.norm_v(v.flatten(0, 1).permute(0, 2, 1)).permute(0, 2, 1).view_as(v)
        attn =  torch.matmul(k.transpose(-2, -1), v) / k.shape[-2]
        out = torch.matmul(q, attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MyTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        attn_norm_type='ln',
        dim_head = 64,
        heads = 8,
        layer_norm=True,
        norm_eps=1e-5,
        mlp_hidden_dim=None,
    ):
        super().__init__()
        self.attn = MyAttention(dim=dim,
                                norm_type=attn_norm_type,
                                dim_head=dim_head,
                                heads=heads)
        
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps) if layer_norm else nn.Identity()
        self.mlp = Mlp(dim,
                       hidden_features=mlp_hidden_dim)
    
    def forward(
        self,
        x,
    ):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, causal=False, heads=heads, dim_head=dim_head) #Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        # y = self.attn(x, mask) #
        # if return_attention:
        #     return attn
        # x = self.norm1(x + self.drop_path(y))
        # x = self.norm2(x + self.drop_path(self.mlp(x)))


        y = self.attn(self.norm1(x), mask) #
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MyTransformer(nn.Module):
    def __init__(self, num_layers, num_features, k, norm_type, act_type, d_backbone, normalize_over=[0, 1], momentum=0.99, projector=False, upsample_factor=1):
        super().__init__()
        self.num_layers = num_layers
        self.out_features = k
        if self.num_layers == 0:
            self.out_features = d_backbone
            return

        self.is_transformer = True
        self.normalize_over = normalize_over
        self.momentum = momentum
        self.num_features = d_backbone*3
        self.projector = projector
        self.upsample_factor = upsample_factor
        
        # self.preprocesser = nn.Sequential(FeedForward(d_backbone, d_backbone, 0), nn.LayerNorm(d_backbone))
        # self.preprocesser2 = nn.Sequential(nn.Linear(d_backbone * 2, d_backbone))

        layers = []
        for i in range(num_layers - 1):
            # layers.append(MyTransformerLayer(dim=d_backbone, attn_norm_type=norm_type, layer_norm= act_type=='postln'))
            layers.append(Block(d_backbone*3, d_backbone*3//64, 64, num_features, 0, 0))
        layers.append(nn.LayerNorm(d_backbone*3))
        self.fn = nn.Sequential(*layers)
        self.head = nn.Linear(d_backbone*3, k * (self.upsample_factor ** 2) * 2, bias=True)

        # self.shortcut = (nn.Linear(sizes[0], sizes[-1]))

        if self.projector:
            assert False
            self.fn[-1] = nn.Identity()
            sizes = [num_features, 8192, self.out_features]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
            self.projector_fn = nn.Sequential(*layers)

        self.register_buffer('eigennorm', torch.zeros(k * 2))
        self.register_buffer('num_calls', torch.Tensor([0]))

        # self.apply(init_weights)
    
        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     # we use xavier_uniform following official JAX ViT:
        #     torch.nn.init.xavier_uniform_(m.weight)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, (nn.LayerNorm)):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

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

    def forward(self, x):
        if self.num_layers == 0:
            return x, x

        # hidden = self.fn(self.preprocesser(lowlevel_x) + x)# + self.shortcut(x)
        hidden = self.fn(x)# + self.shortcut(x)
        ret_raw = self.head(hidden)
        ret_raw = rearrange(ret_raw, "b (w h) (c d l) -> b (w c) (h d) l", w=int(math.sqrt(ret_raw.shape[1])), c=self.upsample_factor, d=self.upsample_factor).flatten(1, 2)

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
        
        if self.projector:
            return hidden, ret_raw / norm_
        else:
            ret = ret_raw / norm_
            return hidden, ret