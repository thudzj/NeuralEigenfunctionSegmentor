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
        psi,
        n_cls,
        kmeans_cfg,
        neuralef_loss_cfg,
        backbone_trained_by_dino=False,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = backbone.patch_embed.patch_size
        self.backbone = backbone
        self.psi = psi

        self.kmeans_cfg = kmeans_cfg
        self.kmeans_n_cls = kmeans_cfg['n_cls']
        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.backbone_trained_by_dino = backbone_trained_by_dino
  
        freeze_all_layers_(self.backbone)

        self.feat_out = {}
        self.backbone._modules["blocks"][0].register_forward_hook(self.hook_fn_forward_lowlevel)
        self.backbone._modules["blocks"][len(self.backbone._modules["blocks"]) // 2].register_forward_hook(self.hook_fn_forward_midlevel)
        if backbone_trained_by_dino:
            self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_highlevel)

        self.clustering_feature_dim = self.forward(torch.zeros(1, 3, 512, 512), return_features=True).shape[-1]
        self.online_head = nn.Linear(self.clustering_feature_dim, n_cls)
        # self.online_head2 = nn.Linear(self.backbone.d_model, n_cls)

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
            onehot_assignments = F.one_hot(assignments, self.kmeans_n_cls)
            self.cluster_centers.mul_(self.num_per_cluster.view(-1, 1)).add_(onehot_assignments.float().T @ feature1)
            self.num_per_cluster.add_(onehot_assignments.long().sum(0))
            self.cluster_centers.div_(self.num_per_cluster.view(-1, 1))
        return logits.view(*feature.shape[:-1], -1).div_(self.kmeans_cfg['tau']) #onehot_assignments.float().view(*feature.shape[:-1], -1) #
    
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

    def forward(self, im, return_neuralef_loss=False, return_features=False):
        bs = im.shape[0]

        with torch.no_grad():
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            h = H // self.patch_size
            
        hidden, Psi = self.psi(torch.cat([lowlevel_feature, midlevel_feature, highlevel_feature], dim=-1))

        with torch.no_grad():
            if self.kmeans_cfg['feature'] == 'original_feature':
                clustering_feature = highlevel_feature
            elif self.kmeans_cfg['feature'] == 'original_all_feature':
                clustering_feature = torch.cat([lowlevel_feature, midlevel_feature, highlevel_feature], -1)
            elif self.kmeans_cfg['feature'] == 'normalized_original_all_feature':
                clustering_feature = torch.cat([F.normalize(lowlevel_feature, dim=-1), F.normalize(midlevel_feature, dim=-1), F.normalize(highlevel_feature, dim=-1)], -1)
            elif self.kmeans_cfg['feature'] == 'hidden_feature':
                clustering_feature = hidden.clone().detach()
            elif self.kmeans_cfg['feature'] == 'eigen_feature':
                clustering_feature = Psi.clone().detach()
            elif self.kmeans_cfg['feature'] == 'original&hidden_feature':
                clustering_feature = torch.cat([hidden.clone().detach(), highlevel_feature], -1)
            elif self.kmeans_cfg['feature'] == 'original&eigen_feature':
                if Psi.shape[1] == highlevel_feature.shape[1]:
                    clustering_feature = torch.cat([Psi.clone().detach(), highlevel_feature], -1)
                else:
                    clustering_feature = highlevel_feature
            elif self.kmeans_cfg['feature'] == 'normalized_original_feature':
                clustering_feature = F.normalize(highlevel_feature, dim=-1)
            elif self.kmeans_cfg['feature'] == 'normalized_hidden_feature':
                clustering_feature = F.normalize(hidden.clone().detach(), dim=-1)
            elif self.kmeans_cfg['feature'] == 'normalized_original&hidden_feature':
                clustering_feature = torch.cat([F.normalize(hidden.clone().detach(), dim=-1) * math.sqrt(hidden.shape[-1]), 
                                                F.normalize(highlevel_feature, dim=-1) * math.sqrt(highlevel_feature.shape[-1])], -1)
            elif self.kmeans_cfg['feature'] == 'normalized_original&eigen_feature':
                if Psi.shape[1] == highlevel_feature.shape[1]:
                    clustering_feature = torch.cat([F.normalize(Psi.clone().detach(), dim=-1), F.normalize(highlevel_feature, dim=-1)], -1)
                else:
                    clustering_feature = highlevel_feature
            else:
                assert False, self.kmeans_cfg['feature']
            
            clustering_feature = rearrange(clustering_feature, "b (h w) c -> b h w c", h=h)
            if not self.training and hasattr(self, 'proj_matrix'):
                clustering_feature = clustering_feature.sub(self.feature_mean) @ self.proj_matrix

        if return_features:
            return clustering_feature #F.normalize(clustering_feature, dim=-1)

        if not self.training and hasattr(self, 'cluster_centers'):
            masks = self.clustering(clustering_feature).permute(0, 3, 1, 2) #F.normalize(clustering_feature, dim=-1)
        else:
            masks = self.online_head(clustering_feature).permute(0, 3, 1, 2)
        upsample_bs = 16 if self.training else 2
        masks = torch.cat([F.interpolate(masks[i*upsample_bs:min((i+1)*upsample_bs, len(masks))], size=(H, W), mode="bilinear")
            for i in range(int(math.ceil(float(len(masks))/upsample_bs)))])
        masks = unpadding(masks, (H_ori, W_ori))

        if return_neuralef_loss:
            # masks2 = self.online_head2(highlevel_feature).view(*clustering_feature.shape[:-1], -1).permute(0, 3, 1, 2)
            # upsample_bs = 16 if self.training else 2
            # masks2 = torch.cat([F.interpolate(masks2[i*upsample_bs:min((i+1)*upsample_bs, len(masks2))], size=(H, W), mode="bilinear")
            #     for i in range(int(math.ceil(float(len(masks2))/upsample_bs)))])
            # masks2 = unpadding(masks2, (H_ori, W_ori))

            with torch.no_grad():
                im_ = F.interpolate(im, size=(H // self.patch_size, W // self.patch_size), mode="bilinear")
            neuralef_loss, neuralef_reg = cal_neuralef_loss(highlevel_feature, Psi, im_, self.neuralef_loss_cfg)
            return masks, neuralef_loss, neuralef_reg
        return masks


def cal_neuralef_loss(highlevel_feature, Psi, im, neuralef_loss_cfg):
    if Psi.dim() == 3:
        Psi = Psi.flatten(0, 1)
    Psi *= math.sqrt(neuralef_loss_cfg['t'] / Psi.shape[0])
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            highlevel_feature_ = F.normalize(highlevel_feature.flatten(0, 1), dim=-1)
            A = highlevel_feature_ @ highlevel_feature_.transpose(-1, -2)
            A.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
            A.clamp_(min=0.)
            ret = torch.topk(A, neuralef_loss_cfg['num_nearestn_feature'], dim=-1)
            res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
            res.scatter_(-1, ret.indices, ret.values)
            res.add_(res.T).div_(2.)
            A = res 

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

    Psi1, Psi2 = Psi.chunk(2, dim=-1)
    loss, reg = 0, 0
    for Psi_, gram_matrix_, weight_ in zip([Psi1, Psi2], [gram_matrix, gram_matrix2], [1, neuralef_loss_cfg['pixelwise_weight']]):
        R = Psi_.T @ gram_matrix_ @ Psi_
        if neuralef_loss_cfg['no_sg']:
            R_hat = R
        else:
            R_hat = Psi_.T.detach() @ gram_matrix_ @ Psi_
        loss += - R.diagonal().sum() / Psi.shape[1] * weight_
        reg += (R_hat ** 2).triu(1).sum() / Psi.shape[1] * weight_
    return loss, reg * neuralef_loss_cfg['alpha']
