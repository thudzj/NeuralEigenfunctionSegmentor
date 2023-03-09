import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from kornia.filters import canny

from model.utils import padding, unpadding
from einops import rearrange
from utils.torch import freeze_all_layers_

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


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
        self.feature_cov = None
        self.feature_count = 0
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

        self.online_head = nn.Linear(self.encoder.embedding_dim, n_cls)

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
        #print("highlevel_features shape",highlevel_feature.shape) #highlevel_features shape torch.Size([16, 900, 384])
        if self.backbone_trained_by_dino:
            highlevel_feature = self.feat_out["highlevel_feature"]
        lowlevel_feature = self.feat_out["lowlevel_feature"]
        midlevel_feature = self.feat_out["midlevel_feature"]
        return lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W

    def forward(self, im, return_all=False):
        with torch.no_grad():
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            h = H // self.patch_size
            if self.feature_cov is None:
                F.normalize(highlevel_feature, p=2, dim=2)
                self.feature_count += highlevel_feature.shape[0]*highlevel_feature.shape[1]
                self.feature_cov = torch.einsum("blk,blm -> km",F.normalize(highlevel_feature, p=2, dim=2),F.normalize(highlevel_feature, p=2, dim=2))
                self.feature_cov /= self.feature_count
            else:
                self.feature_cov = self.feature_cov * self.feature_count + torch.einsum("blk,blm -> km",F.normalize(highlevel_feature, p=2, dim=2),F.normalize(highlevel_feature, p=2, dim=2))
                self.feature_count += highlevel_feature.shape[0]*highlevel_feature.shape[1]
                self.feature_cov /= self.feature_count

        features_map = {"high": highlevel_feature, "mid": midlevel_feature, "low": midlevel_feature}
        x = torch.cat([features_map[item] for item in self.encoder.inputs], dim=-1)
        #x = torch.cat([x, hog_feature], dim =-1)
        z_e_x, z_q_x_st, z_q_x, vq_logits, distance_loss = self.encoder(x)
        
        if self.training: 
            logits = self.online_head(z_e_x.detach().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # print("logits shape",logits.shape)#torch.Size([16, 60, 480, 480])
        else:
            logits = vq_logits
        
        # if logits.shape[2] != H:
        #     logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        logits = unpadding(logits, (H_ori, W_ori))

        #entropy loss
        entropy = F.softmax(torch.abs(vq_logits).view(-1), dim=0) * F.log_softmax(torch.abs(vq_logits).view(-1), dim=0)
        entropy_loss = -1.0 * entropy.sum(0)

        if return_all:
            x_tilde = self.decoder(z_q_x_st, highlevel_feature)
            #print("s_tild.shape",x_tilde.shape)
            if self.loss_cfg['recon_target'] == 'highlevel_feature':
                recon_loss = F.mse_loss(x_tilde, highlevel_feature)
            elif self.loss_cfg['recon_target'] == 'midlevel_feature':
                recon_loss = F.mse_loss(x_tilde, midlevel_feature)
            elif self.loss_cfg['recon_target'] == 'lowlevel_feature':
                recon_loss = F.mse_loss(x_tilde, lowlevel_feature)
            elif self.loss_cfg['recon_target'] == 'canny_feature':
                recon_loss = F.mse_loss(x_tilde, canny_feature)
            elif self.loss_cfg['recon_target'] == 'pca_feature':
                l,v = torch.linalg.eigh(self.feature_cov)
                #print(l[:10])
                #u,_,v = torch.pca_lowrank(highlevel_feature,center=False,q=64)
                rec_highlevel_feature = torch.matmul(highlevel_feature,v[:,-16:])
                recon_loss = F.mse_loss(x_tilde, rec_highlevel_feature)
            # elif self.loss_cfg['recon_target'] == 'hog_feature':
            #     recon_loss = F.mse_loss(x_tilde, hog_feature)
            elif self.loss_cfg['recon_target'] == 'raw_pixels':
                raise NotImplementedError
            else:
                raise NotImplementedError
            recon_loss = recon_loss
            vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.loss_cfg['beta']
            return logits, vq_logits, recon_loss, vq_loss, commit_loss, distance_loss, entropy_loss
        return logits