import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from einops import rearrange

from utils.torch import freeze_all_layers_
from kmeans_pytorch import kmeans

class Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        psi,
        n_cls,
        kmeans_cfg,
        neuralef_loss_cfg,
        feature_mean_momentum=0.99
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = backbone.patch_embed.patch_size
        self.backbone = backbone
        self.psi = psi
        self.kmeans_cfg = kmeans_cfg
        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.feature_mean_momentum = feature_mean_momentum
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.register_buffer("num_per_cluster", torch.zeros(n_cls))
        self.register_buffer("cluster_centers", torch.zeros(n_cls, self.psi.fn[-1].out_features))
        self.register_buffer("feature_mean", torch.zeros(self.backbone.d_model))

        freeze_all_layers_(self.backbone)
    

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("backbone.", self.backbone).union(
            append_prefix_no_weight_decay("psi.", self.psi)
        )
        return nwd_params

    @torch.no_grad()
    def clustering(self, Psi):
        # l2 normalize over the feature dim
        Psi = F.normalize(Psi, dim=1)

        if self.num_calls == 0:
            assert self.training, "we should train psi and then perform inference"
            assignments, cluster_centers = kmeans(
                X=Psi, num_clusters=self.n_cls, distance='euclidean', device=Psi.device)
            self.cluster_centers.copy_(cluster_centers)
            onehot_assignments = F.one_hot(assignments, self.n_cls)
            self.num_per_cluster.copy_(onehot_assignments.long().sum(0))

        # Eucelidean distance
        logits = (Psi ** 2).sum(1).view(-1, 1) + (self.cluster_centers ** 2).sum(1).view(1, -1) - 2 * Psi @ self.cluster_centers.T
        logits = -logits
    
        if self.num_calls > 0 and self.training:
            assignments = logits.argmax(dim=1)
            onehot_assignments = F.one_hot(assignments, self.n_cls)
            self.num_per_cluster.add_(onehot_assignments.long().sum(0))
            if isinstance(self.kmeans_cfg['momentum'], float):
                momentum = self.kmeans_cfg['momentum'] * torch.ones(self.n_cls, device=Psi.device)
            else:
                assert self.kmeans_cfg['momentum'] == 'auto'
                momentum = 1 - 1. / self.num_per_cluster
            self.cluster_centers.mul_(momentum.view(-1, 1)).add_(onehot_assignments.float().T @ Psi * (1 - momentum.view(-1, 1)))
        return logits

    def forward(self, im, return_neuralef_loss=False):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.backbone.forward(im, return_features=True)

        with torch.no_grad():
            if self.num_calls == 0:
                self.feature_mean.copy_(x.mean((0, 1)))
            else:
                self.feature_mean.mul_(self.feature_mean_momentum).add_((1 - self.feature_mean_momentum) * x.mean((0, 1)))

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.backbone.distilled
        x = x[:, num_extra_tokens:]
        FF = x.flatten(0, 1)
        Psi = self.psi(FF)

        masks = self.clustering(Psi).view(x.shape[0], x.shape[1], self.n_cls)
        masks = rearrange(masks, "b (h w) c -> b c h w", h=H // self.patch_size)
        masks = torch.cat([F.interpolate(masks[i*16:min((i+1)*16, len(masks))], size=(H, W), mode="bilinear")
            for i in range(int(math.ceil(float(len(masks))/16.)))])
        masks = unpadding(masks, (H_ori, W_ori))

        if self.training:
            self.num_calls += 1

        if return_neuralef_loss:
            neuralef_loss, neuralef_reg = cal_neuralef_loss(FF, Psi, self.feature_mean, self.neuralef_loss_cfg)
            return masks, neuralef_loss, neuralef_reg
       
        return masks


def cal_neuralef_loss(FF, Psi, feature_mean, neuralef_loss_cfg):
    Psi *= neuralef_loss_cfg['t'] / Psi.shape[0]
    
    if neuralef_loss_cfg['input_l2_normalize'] == True:
        FF = F.normalize(FF, dim=1)
        feature_mean = F.normalize(feature_mean, dim=-1)

    with torch.no_grad():
        if neuralef_loss_cfg['kernel'] == 'normalized_adjacency':
            D = FF @ feature_mean
            if neuralef_loss_cfg['mask_neg']:
                D *= (D >= 0)
            D = D.rsqrt()
            # print(D[:20])
            D[D == float('inf')] = 0
            D[D == float('-inf')] = 0
            D[D == float('nan')] = 0
            D = torch.diag(D)
            gram_matrix = D @ FF @ FF.T @ D # /  neuralef_loss_cfg['num']
            # if neuralef_loss_cfg['mask_neg']:
            #     gram_matrix *= (gram_matrix >= 0).float()
            # gram_matrix.diagonal().add_(1)
            # print(gram_matrix[:5, :5])
        elif neuralef_loss_cfg['kernel'] == 'per_batch_normalized_adjacency':
            A = FF @ FF.T
            D = A.sum(-1)
            if neuralef_loss_cfg['mask_neg']:
                D *= (D >= 0)
            D = D.rsqrt()
            D[D == float('inf')] = 0
            D[D == float('-inf')] = 0
            D[D == float('nan')] = 0
            D = torch.diag(D)
            gram_matrix = D @ A @ D
            # gram_matrix.diagonal().add_(1)
        elif neuralef_loss_cfg['kernel'] == 'linear':
            gram_matrix = FF @ FF.T
        else:
            assert False

    R = Psi.T @ gram_matrix @ Psi
    R_hat = Psi.T.detach() @ gram_matrix @ Psi
    loss = - R.diagonal().sum() * 2
    reg = (R_hat ** 2).triu(1).sum()
    return loss, reg * neuralef_loss_cfg['alpha']
