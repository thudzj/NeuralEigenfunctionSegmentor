import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math

############################################################
############### for vector quantization ####################
############################################################

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = VectorQuantization.apply(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = VectorQuantization.apply(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = VectorQuantizationStraightThrough.apply(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_size = z_e_x_.size()
            inputs_flatten = z_e_x_.view(-1, embedding_size)

            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, self.embedding.weight.t(), alpha=-2.0, beta=1.0)
            distances = distances.view(*inputs_size[:-1],self.embedding.weight.size(0))
            logits = -distances.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, logits

############################################################
############## for convolutional encoder  ##################
############################################################

from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_
from model.blocks import FeedForward
from einops import rearrange, repeat

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
            n_embeddings=128
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

        self.initialize_weights()
        self.codebook = VQEmbedding(n_embeddings, embedding_dim)


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

        z_q_x_st, z_q_x, logits = self.codebook.straight_through(z_e_x)
        return z_e_x, z_q_x_st, z_q_x, logits

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
