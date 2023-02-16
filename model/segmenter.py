import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from einops import rearrange
from utils.torch import freeze_all_layers_


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
        if self.backbone_trained_by_dino:
            highlevel_feature = self.feat_out["highlevel_feature"]
        lowlevel_feature = self.feat_out["lowlevel_feature"]
        midlevel_feature = self.feat_out["midlevel_feature"]
        return lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W

    def forward(self, im, return_all=False):
        with torch.no_grad():
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            h = H // self.patch_size
        
        features_map = {"high": highlevel_feature, "mid": midlevel_feature, "low": midlevel_feature}
        x = torch.cat([features_map[item] for item in self.encoder.inputs], dim=-1)
        z_e_x, z_q_x_st, z_q_x, vq_logits = self.encoder(x)
        
        if self.training:
            logits = self.online_head(z_e_x.detach().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            logits = vq_logits
        
        # if logits.shape[2] != H:
        #     logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        logits = unpadding(logits, (H_ori, W_ori))

        if return_all:
            x_tilde = self.decoder(z_q_x_st)
            if self.loss_cfg['recon_target'] == 'highlevel_feature':
                recon_loss = F.mse_loss(x_tilde, highlevel_feature)
            elif self.loss_cfg['recon_target'] == 'midlevel_feature':
                recon_loss = F.mse_loss(x_tilde, midlevel_feature)
            elif self.loss_cfg['recon_target'] == 'lowlevel_feature':
                recon_loss = F.mse_loss(x_tilde, lowlevel_feature)
            elif self.loss_cfg['recon_target'] == 'raw_pixels':
                raise NotImplementedError
            else:
                raise NotImplementedError

            vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
            commit_loss = F.mse_loss(z_e_x, z_q_x.detach()) * self.loss_cfg['beta']
            return logits, vq_logits, recon_loss, vq_loss, commit_loss

        return logits

