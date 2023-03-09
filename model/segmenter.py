import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kornia.filters import canny,gaussian_blur2d

import torchvision
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

from model.utils import padding, unpadding
from einops import rearrange
from utils.torch import freeze_all_layers_

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig("canny.png",dpi=600) 

class Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        encoder, 
        decoder,
        n_cls,
        loss_cfg,
        backbone_trained_by_dino=False,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = backbone.patch_embed.patch_size
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.loss_cfg = loss_cfg
        self.backbone_trained_by_dino = backbone_trained_by_dino
  
        freeze_all_layers_(self.backbone)

        #use HOG feature
        self.feat_layer = HOG(nbins=9, pool=8, gaussian_window=16)

        self.feat_out = {}
        self.backbone._modules["blocks"][0].register_forward_hook(self.hook_fn_forward_lowlevel)
        self.backbone._modules["blocks"][len(self.backbone._modules["blocks"]) // 2].register_forward_hook(self.hook_fn_forward_midlevel)
        if backbone_trained_by_dino:
            self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_highlevel)

        self.online_head = nn.Linear(self.encoder.n_embeddings, n_cls)

    def hook_fn_forward_lowlevel(self, module, input, output):
        self.feat_out["lowlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_midlevel(self, module, input, output):
        self.feat_out["midlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_highlevel(self, module, input, output):
        output_qkv = output.reshape(output.shape[0], output.shape[1], 3, self.backbone._modules["blocks"][-1]._modules["attn"].heads, -1).permute(2, 0, 3, 1, 4)
        self.feat_out["highlevel_feature"] = output_qkv[1].transpose(1, 2).reshape(output.shape[0], output.shape[1], -1)[:, 1 + self.backbone.distilled:, :]

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("backbone.", self.backbone).union(
            append_prefix_no_weight_decay("encoder.", self.encoder)).union(
            append_prefix_no_weight_decay("decoder.", self.decoder))
        return nwd_params

    def forward_features(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        highlevel_feature = self.backbone.forward(im, return_features=True)[:, 1 + self.backbone.distilled:, :]

        if self.backbone_trained_by_dino:
            highlevel_feature = self.feat_out["highlevel_feature"]
        lowlevel_feature = self.feat_out["lowlevel_feature"]
        midlevel_feature = self.feat_out["midlevel_feature"]
        return lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W

    def forward(self, im, return_all=False, tau=None):
        with torch.no_grad():
            #im = gaussian_blur2d(im, (3, 3), (1.5, 1.5))
            # gaussian_im = gaussian_blur2d(im, (3, 3), (1.5, 1.5))
            # print(im[0]) 和gaussian也是-1到1
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            h = H // self.patch_size

        features_map = {"high": highlevel_feature, "mid": midlevel_feature, "low": midlevel_feature}
        x = torch.cat([features_map[item] for item in self.encoder.inputs], dim=-1)
        #x = torch.cat([x, hog_feature], dim =-1)
        z_e_x, z_q_x = self.encoder(x, tau)

        cos = lambda m: F.normalize(m, dim=-1) @ F.normalize(m, dim=-1).t()
        # feature_sim = torch.stack([cos(m) for m in highlevel_feature])  # [16, 900, 900]
        # print(feature_sim.shape)
        
        if self.training: 
            logits = self.online_head(z_e_x.detach().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # print("logits shape",logits.shape)#torch.Size([16, 60, 480, 480])
        else:
            logits = z_e_x
        
        # if logits.shape[2] != H:
        #     logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        logits = unpadding(logits, (H_ori, W_ori))

        # entropy loss
        # entropy = F.softmax(torch.abs(vq_logits).view(-1), dim=0) * F.log_softmax(torch.abs(vq_logits).view(-1), dim=0)
        # entropy_loss = -1.0 * entropy.sum(0)

        if return_all:
            x_tilde = self.decoder(z_q_x)
            if self.loss_cfg['recon_target'] == 'highlevel_feature':
                recon_loss = F.mse_loss(x_tilde, highlevel_feature)
            elif self.loss_cfg['recon_target'] == 'midlevel_feature':
                recon_loss = F.mse_loss(x_tilde, midlevel_feature)
            elif self.loss_cfg['recon_target'] == 'lowlevel_feature':
                recon_loss = F.mse_loss(x_tilde, lowlevel_feature)
            elif self.loss_cfg['recon_target'] == 'canny_feature':
                recon_loss = F.mse_loss(x_tilde, canny_feature)
            elif self.loss_cfg['recon_target'] == 'gaussian_im':
                gaussian_im = rearrange(gaussian_im, "b c (w p) (h p1) -> b (w h) (c p p1)", w=int(gaussian_im.shape[2]/16),h=int(gaussian_im.shape[2]/16))
                recon_loss = F.mse_loss(x_tilde, gaussian_im)
            elif self.loss_cfg['recon_target'] == 'pca_feature':
                u,_,_ = torch.pca_lowrank(highlevel_feature,q=64)
                recon_loss = F.mse_loss(x_tilde, u)
            elif self.loss_cfg['recon_target'] == 'similarity':
                cos = lambda m: F.normalize(m,dim=-1) @ F.normalize(m,dim=-1).t()
                # x_tilde_sim = torch.stack([cos(m) for m in x_tilde])
                x_tilde = rearrange(x_tilde, "b (w h) c -> (b w h) c", w=int(math.sqrt(x_tilde.shape[1])))
                x_tilde_sim = cos(x_tilde)
                highlevel_feature = rearrange(highlevel_feature, "b (w h) c -> (b w h) c", w=int(math.sqrt(highlevel_feature.shape[1])))
                feature_sim = cos(highlevel_feature)
                recon_loss = F.mse_loss(x_tilde_sim, feature_sim)
            # elif self.loss_cfg['recon_target'] == 'hog_feature':
            #     recon_loss = F.mse_loss(x_tilde, hog_feature)
            elif self.loss_cfg['recon_target'] == 'raw_pixels':
                raise NotImplementedError
            else:
                raise NotImplementedError
            recon_loss = recon_loss # * self.loss_cfg['rec_ratio']
            # vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            # commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.loss_cfg['beta']
            return logits, z_e_x, recon_loss #, vq_loss, commit_loss, distance_loss
        return logits
