import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math


############################################################
############## for convolutional encoder  ##################
############################################################

from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_
from model.blocks import FeedForward
from einops import rearrange, repeat
from torch.nn.utils.parametrizations import orthogonal

#transformer layer
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

class Encoder(nn.Module):
    def __init__(self, 
            input_type, 
            d_backbone, 
            patch_size, 
            d_model, 
            mlp_dim, 
            n_layers=2, 
            num_heads=None, 
            head_dim=64, 
            upsample_ratio=1, 
            embedding_dim=32, 
            n_embeddings=128,
            orthogonal_embedding=True,
            orthogonal_embedding_dim=512
        ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.inputs = list(input_type.split("-"))

        if num_heads is None:
            num_heads = d_model//head_dim

        self.preprocess = nn.Linear(d_backbone * len(self.inputs), d_model)
        blocks = []
        for i in range(n_layers):
            blocks.append(Block(d_model, num_heads, head_dim, mlp_dim))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, embedding_dim, bias=True)

        self.upsample_ratio = upsample_ratio
        self.remaining_upsample_ratio = patch_size // (upsample_ratio ** n_layers)

        if orthogonal_embedding:
            self.embeddings = orthogonal(nn.Linear(orthogonal_embedding_dim, n_embeddings, bias=False))
        else:
            self.embeddings = nn.Linear(orthogonal_embedding_dim, n_embeddings, bias=False)
        self.register_parameter("key_weight_for_embeddings", nn.Parameter(trunc_normal_(torch.randn(embedding_dim, orthogonal_embedding_dim), std=0.02)))
        self.register_parameter("value_weight_for_embeddings", nn.Parameter(trunc_normal_(torch.randn(embedding_dim, orthogonal_embedding_dim), std=0.02)))

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

    def forward(self, x):
        out = self.preprocess(x)
        for block in self.blocks:
            out = block(out)
            if self.upsample_ratio > 1:
                out = rearrange(out, "b (w h) l -> b l w h", w=int(math.sqrt(out.shape[1])))
                out = F.interpolate(out, size=(out.shape[2] * self.upsample_ratio, out.shape[3] * self.upsample_ratio), mode="bilinear")
                out = out.permute(0, 2, 3, 1).flatten(1, 2)

        out = self.head(self.norm(out))
        out = rearrange(out, "b (w h) l -> b l w h", w=int(math.sqrt(out.shape[1])))
        z_e_x = F.interpolate(out, size=(out.shape[2] * self.remaining_upsample_ratio, out.shape[3] * self.remaining_upsample_ratio), mode="bilinear")

        key = self.embeddings(self.key_weight_for_embeddings).T
        value = self.embeddings(self.value_weight_for_embeddings).T

        logits = torch.einsum("bdwh,ld->blwh", z_e_x, key)
        z_after_attn = torch.einsum("blwh,ld->bdwh", logits.softmax(1), value)
        return z_e_x, logits, z_after_attn

############################################################
############## for transformer-based decoder  ##############
############################################################

from model.vit import PatchEmbedding
from model.utils import init_weights, resize_pos_embed

class Decoder(nn.Module):
    def __init__(self,
            d_backbone,
            patch_size,
            image_size,
            d_model,
            mlp_dim, 
            n_layers=2, 
            num_heads=None, 
            head_dim=64,
            embedding_dim=32,
            apply_pos_embed=False,
        ):
        super().__init__()

        if num_heads is None:
            num_heads = d_model//head_dim
        self.apply_pos_embed = apply_pos_embed

        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            embedding_dim,
        )
        # pos tokens
        if apply_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches, d_model)
            )

        blocks = []
        for i in range(n_layers):
            blocks.append(Block(d_model, num_heads, head_dim, mlp_dim))
        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_backbone, bias=True)

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

    def forward(self, im):
        _, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        if self.apply_pos_embed:
            pos_embed = self.pos_embed
            if x.shape[1] != pos_embed.shape[1]:
                pos_embed = resize_pos_embed(
                    pos_embed,
                    self.patch_embed.grid_size,
                    (H // PS, W // PS),
                    0,
                )
            x = x + pos_embed
        return self.head(self.norm(self.blocks(x)))
