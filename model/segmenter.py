import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from einops import rearrange
from utils.torch import freeze_all_layers_
from model.vit import VisionTransformer

class Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        psi,
        n_cls,
        neuralef_loss_cfg,
        backbone_trained_by_dino=False,
    ):
        super().__init__()
        if isinstance(backbone, VisionTransformer):
            self.backbone_from_clip = False
        else:
            self.backbone_from_clip = True
            backbone.distilled = 0
        self.n_cls = n_cls
        self.patch_size = backbone.conv1.kernel_size[0] if self.backbone_from_clip else backbone.patch_embed.patch_size
        self.backbone = backbone
        self.psi = psi

        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.backbone_trained_by_dino = backbone_trained_by_dino
  
        freeze_all_layers_(self.backbone)

        self.feat_out = {}
        if self.backbone_from_clip:
            self.backbone.transformer.resblocks[0].register_forward_hook(self.hook_fn_forward_lowlevel_clip)
            self.backbone.transformer.resblocks[len(self.backbone.transformer.resblocks) // 2].register_forward_hook(self.hook_fn_forward_midlevel_clip)
        else:
            self.backbone._modules["blocks"][0].register_forward_hook(self.hook_fn_forward_lowlevel)
            self.backbone._modules["blocks"][len(self.backbone._modules["blocks"]) // 2].register_forward_hook(self.hook_fn_forward_midlevel)
            if backbone_trained_by_dino:
                self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_highlevel)

        self.online_head = nn.Linear(self.psi.hidden_dim, n_cls)

        self.mode = 'our'

    def hook_fn_forward_lowlevel(self, module, input, output):
        self.feat_out["lowlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_midlevel(self, module, input, output):
        self.feat_out["midlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_highlevel(self, module, input, output):
        output_qkv = output.reshape(output.shape[0], output.shape[1], 3, self.backbone._modules["blocks"][-1]._modules["attn"].heads, -1).permute(2, 0, 3, 1, 4)
        self.feat_out["highlevel_feature"] = output_qkv[1].transpose(1, 2).reshape(output.shape[0], output.shape[1], -1)[:, 1 + self.backbone.distilled:, :]

    def hook_fn_forward_lowlevel_clip(self, module, input, output):
        self.feat_out["lowlevel_feature"] = output.permute(1, 0, 2)[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_midlevel_clip(self, module, input, output):
        self.feat_out["midlevel_feature"] = output.permute(1, 0, 2)[:, 1 + self.backbone.distilled:, :]

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("backbone.", self.backbone).union(
            append_prefix_no_weight_decay("psi.", self.psi)
        )
        return nwd_params
    
    @torch.no_grad()
    def clustering(self, feature, update=False):
        feature1 = feature.reshape(-1, feature.shape[-1])

        logits = (feature1 ** 2).sum(1).view(-1, 1) + (self.cluster_centers ** 2).sum(1).view(1, -1) - 2 * feature1 @ self.cluster_centers.T
        logits = -logits
        if update:
            assignments = logits.argmax(dim=1)
            onehot_assignments = F.one_hot(assignments, self.psi.psi_dim)
            self.cluster_centers.mul_(self.num_per_cluster.view(-1, 1)).add_(onehot_assignments.float().T @ feature1)
            self.num_per_cluster.add_(onehot_assignments.long().sum(0))
            self.cluster_centers.div_(self.num_per_cluster.view(-1, 1))
        return logits.view(*feature.shape[:-1], -1)

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

    def forward(self, im, tau=None, return_neuralef_loss=False, return_features=False, none_mask=False):
        oori_h, oori_w = im.shape[2], im.shape[3]
        with torch.no_grad():
            im = F.interpolate(im, self.backbone.input_resolution, mode='bicubic') if self.backbone_from_clip else im
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            h = H // self.patch_size
            
        hidden, Psi = self.psi(torch.cat([lowlevel_feature, midlevel_feature, highlevel_feature], dim=-1), tau)

        if return_features:
            if self.mode == 'our_kmeans':
                return hidden
            elif self.mode == 'kmeans':
                return highlevel_feature
            else:
                assert 0

        if none_mask:
            masks = None
        else:
            if self.training:
                masks = self.online_head(rearrange(hidden, "b (h w) c -> b h w c", h=h)).permute(0, 3, 1, 2)
            else:
                if self.mode == 'our':
                    masks = rearrange(Psi, "b (h w) c -> b h w c", h=h).permute(0, 3, 1, 2).div(self.tau_min)
                elif self.mode == 'our_kmeans':
                    masks = self.clustering(rearrange(hidden, "b (h w) c -> b h w c", h=h)).permute(0, 3, 1, 2)
                elif self.mode == 'kmeans':
                    masks = self.clustering(rearrange(highlevel_feature, "b (h w) c -> b h w c", h=h)).permute(0, 3, 1, 2)
                else:
                    assert 0
            
            upsample_bs = 16 if self.training else 2
            masks = torch.cat([F.interpolate(masks[i*upsample_bs:min((i+1)*upsample_bs, len(masks))], size=(H, W), mode="bilinear")
                for i in range(int(math.ceil(float(len(masks))/upsample_bs)))])
            masks = unpadding(masks, (H_ori, W_ori))

            if self.backbone_from_clip:
                masks = torch.cat([F.interpolate(masks[i*upsample_bs:min((i+1)*upsample_bs, len(masks))], size=(oori_h, oori_w), mode="bilinear")
                    for i in range(int(math.ceil(float(len(masks))/upsample_bs)))])

        if return_neuralef_loss:
            with torch.no_grad():
                im_ = F.interpolate(im, size=(H // self.patch_size, W // self.patch_size), mode="bilinear")
            neuralef_loss, neuralef_reg = cal_neuralef_loss(highlevel_feature, Psi, im_, self.neuralef_loss_cfg)
            return masks, neuralef_loss, neuralef_reg
        return masks


def cal_neuralef_loss(highlevel_feature, Psi, im, neuralef_loss_cfg):
    if Psi.dim() == 3:
        Psi = Psi.flatten(0, 1)
    Psi *= math.sqrt(neuralef_loss_cfg['t'])
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            highlevel_feature_ = F.normalize(highlevel_feature.flatten(0, 1), dim=-1)
            A = highlevel_feature_ @ highlevel_feature_.transpose(-1, -2)
            A.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
            A.clamp_(min=0.)
            ret = torch.topk(A, neuralef_loss_cfg['num_nearestn_feature'], dim=-1)
            res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
            res.scatter_(-1, ret.indices, ret.values)
            A = (res + res.T).div_(2.) 

            im = im.add_(1.).div_(2.)
            bs, h, w = im.shape[0], im.shape[2], im.shape[3]
            x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, -1, 1).repeat(bs, 1, 1)
            y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, -1, 1).repeat(bs, 1, 1)
            im = im.flatten(2).permute(0, 2, 1)
            A_p = None
            for k, distance_weight in zip([neuralef_loss_cfg['num_nearestn_pixel1'], neuralef_loss_cfg['num_nearestn_pixel2']], [2.0, 0.1]):
                if k == 0:
                    continue
                im2 = torch.cat([im, 
                                 x_.mul(distance_weight),
                                 y_.mul(distance_weight)], -1)
                euc_dist = -torch.cdist(im2, im2)
                euc_dist.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                ret = torch.topk(euc_dist, k, dim=2)
                res = torch.zeros(*euc_dist.shape, device=euc_dist.device, dtype=bool)
                res.scatter_(2, ret.indices, torch.ones_like(ret.values).bool())
                if A_p is None:
                    A_p = torch.logical_or(res, res.permute(0, 2, 1))
                else:
                    A_p = torch.logical_or(A_p, torch.logical_or(res, res.permute(0, 2, 1)))
            A2 = torch.block_diag(*A_p.type_as(Psi))

            D = A.sum(-1).rsqrt()
            gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1)).type_as(Psi)

            D2 = A2.sum(-1).rsqrt()
            gram_matrix2 = A2.mul_(D2.view(1, -1)).mul_(D2.view(-1, 1)).type_as(Psi)

            gram_matrix += gram_matrix2 * neuralef_loss_cfg['pixelwise_weight']
    
    R = Psi.T @ gram_matrix @ Psi
    if neuralef_loss_cfg['no_sg']:
        R_hat = R
    else:
        R_hat = Psi.T.detach() @ gram_matrix @ Psi
    loss = - R.diagonal().sum() / Psi.shape[1]
    reg = (R_hat ** 2).triu(1).sum() / Psi.shape[1]
    return loss, reg * neuralef_loss_cfg['alpha']
