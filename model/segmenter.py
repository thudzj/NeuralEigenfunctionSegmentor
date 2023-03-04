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
        #print("img shape",im.shape) #img shape torch.Size([16, 3, 480, 480])
        # hog_feature = self.feat_layer(im) #[16,108,30,30]
        # hog_feature = hog_feature.permute(2, 3, 1, 0)
        # hog_feature = hog_feature.reshape(900,108,-1)
        # hog_feature = hog_feature.permute(2, 0, 1)
        #print("hog_feature shape3",hog_feature.shape) #img_hog shape torch.Size([16,900,108])


        magnitude, edges = canny(im)
        print("edges.shape",edges.shape)
        canny_feature = rearrange(edges, "b c (w p) (h p1) -> b (w h) (c p p1)", w=int(edges.shape[2]/16),h=int(edges.shape[2]/16))

        features_map = {"high": highlevel_feature, "mid": midlevel_feature, "low": midlevel_feature}
        x = torch.cat([features_map[item] for item in self.encoder.inputs], dim=-1)
        #x = torch.cat([x, hog_feature], dim =-1)
        z_e_x, z_q_x_st, z_q_x, vq_logits, distance_loss = self.encoder(x)
        # print(z_q_x_st.shape)#torch.Size([16, 32, 480, 480])
        # print(vq_logits.shape)#torch.Size([16, 128, 480, 480])
        dis = vq_logits.permute(0,2,3,1)
        #print(dis[0][0][0].shape)
        #print(dis[0][0][0])
        
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
            print("s_tild.shape",x_tilde.shape)
            if self.loss_cfg['recon_target'] == 'highlevel_feature':
                recon_loss = F.mse_loss(x_tilde, highlevel_feature)
            elif self.loss_cfg['recon_target'] == 'midlevel_feature':
                recon_loss = F.mse_loss(x_tilde, midlevel_feature)
            elif self.loss_cfg['recon_target'] == 'lowlevel_feature':
                recon_loss = F.mse_loss(x_tilde, lowlevel_feature)
            elif self.loss_cfg['recon_target'] == 'canny_feature':
                recon_loss = F.mse_loss(x_tilde, canny_feature)
            # elif self.loss_cfg['recon_target'] == 'hog_feature':
            #     recon_loss = F.mse_loss(x_tilde, hog_feature)
            elif self.loss_cfg['recon_target'] == 'raw_pixels':
                raise NotImplementedError
            else:
                raise NotImplementedError
            recon_loss = recon_loss * self.loss_cfg['rec_ratio']
            vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.loss_cfg['beta']
            return logits, vq_logits, recon_loss, vq_loss, commit_loss, distance_loss, entropy_loss
        return logits

class HOG(nn.Module):
    """Generate hog feature for each batch images. This module is used in
    Maskfeat to generate hog feature. This code is borrowed from.
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/masked.py>
    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16) -> None:
        super(HOG, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    def _reshape(self, hog_feat: torch.Tensor, hog_h: int) -> torch.Tensor:
        b = hog_feat.shape[0]
        hog_feat = hog_feat.flatten(1, 2)
        unfold_size = hog_feat.shape[-1] // hog_h
        hog_feat = (
            hog_feat.permute(0, 2, 3, 1).unfold(
                1, unfold_size, unfold_size).unfold(
                2, unfold_size, unfold_size).flatten(1, 2).flatten(2))
        return hog_feat.permute(0, 2, 1).reshape(b, -1, hog_h, hog_h)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.
        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out_h = int(h / self.gaussian_window)  # (14, 14)
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = F.normalize(out, p=2, dim=2)

        return self._reshape(out, out_h)

