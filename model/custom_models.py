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

class PlainConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.block(x)

class PlainConvT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim//2, 4, 2, 1),
            nn.BatchNorm2d(dim//2),
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, input_type, d_backbone=768, d_model=512, patch_size=16, embedding_dim=32, n_embeddings=128, residual=True):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.inputs = list(input_type.split("-"))

        num_upsample_times = int(math.log2(patch_size))
        Block = ResBlock if residual else PlainConv
        self.encoder = nn.Sequential(
            nn.Sequential(nn.Conv2d(d_backbone * len(self.inputs), d_model, 3, 1, 1), nn.BatchNorm2d(d_model)),
            *[nn.Sequential(PlainConvT(d_model // (2 ** i)), Block(d_model // (2 ** (i + 1)))) for i in range(num_upsample_times - 1)],
            nn.ReLU(True),
            nn.ConvTranspose2d(d_model // (patch_size // 2), embedding_dim, 4, 2, 1),
        )
        self.codebook = VQEmbedding(n_embeddings, embedding_dim)

        self.apply(self.init_func)

    def init_func(self, m):
        classname = m.__class__.__name__

        if classname.find('Conv2d') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, logits = self.codebook.straight_through(z_e_x)
        return z_e_x, z_q_x_st, z_q_x, logits

############################################################
############## for transformer-based decoder  ##############
############################################################

from timm.models.layers import trunc_normal_
from model.blocks import Block as VitBlock
from model.vit import PatchEmbedding
from model.utils import init_weights, resize_pos_embed

class Decoder(nn.Module):
    def __init__(self,
            d_model,
            n_heads,
            d_ff=None,
            image_size=512,
            patch_size=16,
            embedding_dim=32,
            n_layers=2,
            d_backbone=768
        ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            embedding_dim,
        )

        if d_ff is None:
            mlp_expansion_ratio = 4
            d_ff = mlp_expansion_ratio * d_model

        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model)
        )

        # transformer blocks
        self.blocks = nn.ModuleList(
            [VitBlock(d_model, n_heads, d_ff, 0, 0) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_backbone)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, im):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, num_extra_tokens:]))
