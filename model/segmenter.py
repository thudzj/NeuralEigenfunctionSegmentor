import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from einops import rearrange

from utils.torch import freeze_all_layers_
from sklearn.cluster import SpectralClustering
from fast_pytorch_kmeans import KMeans
import scipy

class Segmenter(nn.Module):
    def __init__(
        self,
        backbone,
        psi,
        n_cls,
        kmeans_cfg,
        neuralef_loss_cfg,
        is_baseline=False,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = backbone.patch_embed.patch_size
        self.backbone = backbone
        self.psi = psi
        self.kmeans_cfg = kmeans_cfg
        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.is_baseline = is_baseline

        # unused buffers
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.register_buffer("feature_mean", torch.zeros(self.backbone.d_model))

        # the clustering buffers
        self.register_buffer("num_per_cluster", torch.zeros(n_cls))
        self.register_buffer("cluster_centers", torch.randn(n_cls, self.psi.fn[-1].out_features))
        
        # a onlien head to check the quality of eigenmaps
        self.online_head = nn.Linear(self.psi.fn[-1].out_features, n_cls)
        
        self.cached_FF = []
        
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
        if self.kmeans_cfg['l2_normalize']:
            Psi = F.normalize(Psi, dim=1)

        # Eucelidean distance
        logits = (Psi ** 2).sum(1).view(-1, 1) + (self.cluster_centers ** 2).sum(1).view(1, -1) - 2 * Psi @ self.cluster_centers.T
        logits = -logits
        assignments = logits.argmax(dim=1)
        onehot_assignments = F.one_hot(assignments, self.n_cls)
    
        if self.training:
            if isinstance(self.kmeans_cfg['momentum'], float):
                momentum = self.kmeans_cfg['momentum'] * torch.ones(self.n_cls, device=Psi.device)
                center_ = onehot_assignments.float().T @ Psi / onehot_assignments.long().sum(0).view(-1, 1)
                self.cluster_centers.mul_(momentum.view(-1, 1)).add_(center_ * (1 - momentum.view(-1, 1)))
                self.num_per_cluster.add_(onehot_assignments.long().sum(0))
            else:
                assert self.kmeans_cfg['momentum'] == 'auto'
                self.cluster_centers.mul_(self.num_per_cluster.view(-1, 1)).add_(onehot_assignments.float().T @ Psi)
                self.num_per_cluster.add_(onehot_assignments.long().sum(0))
                self.cluster_centers.div_(self.num_per_cluster.view(-1, 1))
        return logits #onehot_assignments.float()
    
    @torch.no_grad()
    def baseline_clustering(self, FF, cls_=10, k=32):
        A = FF @ FF.T
        A *= (A >= 0)
        sc = SpectralClustering(n_clusters=cls_, affinity='precomputed').fit(A.data.cpu().numpy())
        logits = F.one_hot(torch.from_numpy(sc.labels_).to(A.device).long(), self.n_cls).float()
        return logits
    
    def forward_features(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.backbone.forward(im, return_features=True)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.backbone.distilled
        x = x[:, num_extra_tokens:]
        return x, im, H_ori, W_ori, H, W

    def forward(self, im, return_neuralef_loss=False, return_eigenmaps=False):
        with torch.no_grad():
            x, im, H_ori, W_ori, H, W = self.forward_features(im)
            x = rearrange(x, "b (h w) c -> b c h w", h=H // self.patch_size)
            if self.neuralef_loss_cfg['upsample_factor'] > 1:
                H_ = self.neuralef_loss_cfg['upsample_factor'] * x.shape[2]
                W_ = self.neuralef_loss_cfg['upsample_factor'] * x.shape[3]
                x = F.interpolate(x, size=(H_, W_), mode="bilinear")
            FF = rearrange(x, "b c h w-> (b h w) c")

            if self.neuralef_loss_cfg['pixelwise_adj_weight'] > 0 and self.training:
                im_ = F.interpolate(im, size=(x.shape[2], x.shape[3]), mode="bilinear")
            else:
                im_ = None
        
        Psi = self.psi(FF)
        if return_eigenmaps:
            return Psi

        if self.is_baseline:
            masks_clustering = self.baseline_clustering(FF).view(x.shape[0], -1, self.n_cls)
        else:
            masks_clustering = None if self.training else self.clustering(Psi).view(x.shape[0], -1, self.n_cls)
        masks_clf = self.online_head(Psi.clone().detach()).view(x.shape[0], -1, self.n_cls)
        
        masks = rearrange(masks_clf if self.training else masks_clustering, "b (h w) c -> b c h w", h=H // self.patch_size*self.neuralef_loss_cfg['upsample_factor'])
        masks = torch.cat([F.interpolate(masks[i*16:min((i+1)*16, len(masks))], size=(H, W), mode="bilinear")
            for i in range(int(math.ceil(float(len(masks))/16.)))])
        masks = unpadding(masks, (H_ori, W_ori))

        if return_neuralef_loss:
            neuralef_loss, neuralef_reg = cal_neuralef_loss(FF, Psi, im_, self.neuralef_loss_cfg, self.cached_FF)

            if len(self.cached_FF) == self.neuralef_loss_cfg["cache_size"]:
                del self.cached_FF[0]
            with torch.no_grad():
                self.cached_FF.append(F.normalize(FF, dim=1) if self.neuralef_loss_cfg['input_l2_normalize'] == True else FF)
            return masks, neuralef_loss, neuralef_reg
       
        return masks


def cal_neuralef_loss(FF, Psi, im, neuralef_loss_cfg, cached_FF):
    Psi *= neuralef_loss_cfg['t'] / Psi.shape[0]
    with torch.no_grad():
        if neuralef_loss_cfg['input_l2_normalize'] == True:
            FF = F.normalize(FF, dim=1)

        if 'normalized_adjacency' in neuralef_loss_cfg['kernel']:
            # the affinity
            A = (FF @ FF.T)
            A.fill_diagonal_(0.)
            if 'thresholded' in neuralef_loss_cfg['kernel']:
                A.clamp_(min=0.)
            
            # combine with pixelwise affinity
            if neuralef_loss_cfg['pixelwise_adj_weight'] > 0:
                with torch.cuda.amp.autocast():
                    im = im.add_(1.).div_(2.)
                    bs, h, w = im.shape[0], im.shape[2], im.shape[3]
                    x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, 1, -1).repeat(bs, 1, 1)
                    y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, 1, -1).repeat(bs, 1, 1)
                    im = im.flatten(2) / neuralef_loss_cfg['pixelwise_adj_div_factor']

                    A_p = 0
                    for k, distance_weight in zip([20, 10], [2.0, 0.1]):
                        im2 = torch.cat([im, 
                                        x_.mul(distance_weight),
                                        y_.mul(distance_weight)], 1)
                        im2_sqr = im2.norm(dim=1)**2
                        euc_dist = 2 * torch.einsum("bcp,bcq->bpq", im2, im2) - im2_sqr[:, :, None] - im2_sqr[:, None, :]
                        euc_dist.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                        ret = torch.topk(euc_dist, k, dim=2)
                        res = torch.zeros_like(euc_dist)
                        res.scatter_(2, ret.indices, torch.ones_like(ret.values))
                        A_p += (res + res.permute(0, 2, 1)) / 4 # we divide by a further 2 because the for loop is of length 2

                for i in range(bs):
                    A[i*(h*w):(i+1)*(h*w), i*(h*w):(i+1)*(h*w)] += neuralef_loss_cfg['pixelwise_adj_weight'] * A_p[i].float()
            
            # estimate D
            if len(cached_FF) == 0:
                D = A.sum(-1) / (A.shape[-1] - 1)
            else:
                b_ = cached_FF[0].shape[0] // 8
                assert cached_FF[0].shape[0] % 8 == 0
                D = sum([(FF @ cached_FF_[i*b_:(i+1)*b_].T).clamp(min=0).sum(-1) for cached_FF_ in cached_FF for i in range(8)]) + A.sum(-1)
                D /= sum([cached_FF_.shape[0] for cached_FF_ in cached_FF]) + A.shape[-1] - 1
            D = D.rsqrt()

            # the gram matrix
            gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1))
            # gram_matrix = gram_matrix.float()
        else:
            assert False

    R = Psi.T @ gram_matrix @ Psi
    if neuralef_loss_cfg['no_sg']:
        R_hat = R
    else:
        R_hat = Psi.T.detach() @ gram_matrix @ Psi
    loss = - R.diagonal().sum()
    reg = (R_hat ** 2).triu(1).sum()
    return loss, reg * neuralef_loss_cfg['alpha']
