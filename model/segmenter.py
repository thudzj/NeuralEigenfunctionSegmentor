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
        feature_mean_momentum=0.99
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = backbone.patch_embed.patch_size
        self.backbone = backbone
        self.psi = psi
        self.kmeans_cfg = kmeans_cfg
        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.is_baseline = is_baseline
        self.feature_mean_momentum = feature_mean_momentum
        self.register_buffer('num_calls', torch.Tensor([0]))
        self.register_buffer("num_per_cluster", torch.zeros(n_cls))
        self.register_buffer("cluster_centers", torch.randn(n_cls, self.psi.fn[-1].out_features))
        self.register_buffer("feature_mean", torch.zeros(self.backbone.d_model))
        self.online_head = nn.Linear(self.psi.fn[-1].out_features, n_cls)
        # self.cached_Psi = []
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
        return x, H_ori, W_ori, H, W

    def eigenmaps(self, im):
        x = self.forward_features(im)[0]
        Psi = self.psi(x.flatten(0, 1))
        return Psi

    def forward(self, im, return_neuralef_loss=False):
        x, H_ori, W_ori, H, W = self.forward_features(im)

        # with torch.no_grad():
        #     if self.neuralef_loss_cfg['input_l2_normalize'] == True:
        #         x_ = F.normalize(x, dim=-1)
        #     else:
        #         x_ = x
        #     feature_mean_ = x_.mean((0, 1))
        #     if self.num_calls == 0:
        #         self.feature_mean.copy_(feature_mean_)
        #     else:
        #         self.feature_mean.mul_(self.feature_mean_momentum).add_((1 - self.feature_mean_momentum) * feature_mean_)

        FF = x.flatten(0, 1)
        Psi = self.psi(FF)

        if self.is_baseline:
            masks_clustering = self.baseline_clustering(FF).view(x.shape[0], x.shape[1], self.n_cls)
        else:
            masks_clustering = self.clustering(Psi).view(x.shape[0], x.shape[1], self.n_cls)
        masks_clf = self.online_head(Psi.clone().detach()).view(x.shape[0], x.shape[1], self.n_cls)
        masks = rearrange(masks_clf if self.training else masks_clustering, "b (h w) c -> b c h w", h=H // self.patch_size)
        masks = torch.cat([F.interpolate(masks[i*16:min((i+1)*16, len(masks))], size=(H, W), mode="bilinear")
            for i in range(int(math.ceil(float(len(masks))/16.)))])
        masks = unpadding(masks, (H_ori, W_ori))

        # if self.training:
        #     self.num_calls += 1

        if return_neuralef_loss:
            neuralef_loss, neuralef_reg = cal_neuralef_loss(FF, Psi, self.feature_mean, self.neuralef_loss_cfg, self.cached_FF)

            if len(self.cached_FF) == 100:
                del self.cached_FF[0]
            with torch.no_grad():
                self.cached_FF.append(F.normalize(FF, dim=1) if self.neuralef_loss_cfg['input_l2_normalize'] == True else FF)
            return masks, neuralef_loss, neuralef_reg
       
        return masks


def cal_neuralef_loss(FF, Psi, feature_mean, neuralef_loss_cfg, cached_FF):
    Psi *= neuralef_loss_cfg['t'] / Psi.shape[0]
    
    if neuralef_loss_cfg['input_l2_normalize'] == True:
        FF = F.normalize(FF, dim=1)

    with torch.no_grad():
        if 'normalized_adjacency' in neuralef_loss_cfg['kernel']:
            A = (FF @ FF.T)
            if not "zero_adj_diag" in neuralef_loss_cfg['kernel']:
                A.fill_diagonal_(0.)
            if 'thresholded' in neuralef_loss_cfg['kernel']:
                A *= (A >= 0)
            if 'per_batch' in neuralef_loss_cfg['kernel']:
                D = A.sum(-1)
                D *= (D >= 0)
                D = D.rsqrt()
                D[D == float('inf')] = 0
                D[D == float('-inf')] = 0
                D[D == float('nan')] = 0
            elif 'cache_for_D' in neuralef_loss_cfg['kernel']:
                if len(cached_FF) == 0:
                    D = A.sum(-1) / (A.shape[-1] - 1)
                else:
                    D = sum([(FF @ cached_FF_.T).clamp(min=0).sum(-1) for cached_FF_ in cached_FF]) + A.sum(-1)
                    D /= sum([cached_FF_.shape[0] for cached_FF_ in cached_FF]) + A.shape[-1] - 1
                D = D.rsqrt()
            else:
                D = FF @ feature_mean
                D *= (D >= 0)
                D = D.rsqrt()
                D[D == float('inf')] = 0
                D[D == float('-inf')] = 0
                D[D == float('nan')] = 0
            gram_matrix = D.view(-1, 1) * A * D.view(1, -1)
            if 'signless' in neuralef_loss_cfg['kernel']:
                gram_matrix.fill_diagonal_(1.)
        elif 'rbf' in neuralef_loss_cfg['kernel']:
            FF_sqr = (FF ** 2).sum(-1)
            # the rbf kernel
            A = FF_sqr.view(-1, 1) + FF_sqr.view(1, -1) - 2 * FF @ FF.T
            sigma2 = float(neuralef_loss_cfg['kernel'].split("_")[1].replace("sigm", ""))
            A = (- A / 2 / sigma2).exp()
            gram_matrix = A
        elif 'custom1' in neuralef_loss_cfg['kernel']:
            A = FF @ FF.T + 1
            A.fill_diagonal_(0.)
            D = FF @ feature_mean + 1
            D = D.rsqrt()
            gram_matrix = D.view(-1, 1) * A * D.view(1, -1)
            if 'signless' in neuralef_loss_cfg['kernel']:
                for term in neuralef_loss_cfg['kernel'].split("_"):
                    if 'signless' in term:
                        break
                gram_matrix.div_(float(term.replace("signless", ""))).fill_diagonal_(1.)
        elif neuralef_loss_cfg['kernel'] == 'linear':
            gram_matrix = FF @ FF.T
        elif neuralef_loss_cfg['kernel'] == 'linear_thresholded':
            gram_matrix = FF @ FF.T
            gram_matrix *= (gram_matrix > 0)
        elif 'nnn' in neuralef_loss_cfg['kernel']:
            knn = int(neuralef_loss_cfg['kernel'].split("_")[0].replace("nnn", ""))
            # the rbf kernel
            A = (FF ** 2).sum(-1).view(-1, 1) + (FF ** 2).sum(-1).view(1, -1) - 2 * FF @ FF.T
            sigma2 = float(neuralef_loss_cfg['kernel'].split("_")[1].replace("sigm", ""))
            A = (- A / 2 / sigma2).exp()
            ret = torch.topk(A, knn, dim=1)
            res = torch.zeros_like(A)
            res.scatter_(1, ret.indices, ret.values)
            gram_matrix = (res + res.T) / 2
            gram_matrix.diagonal().zero_()
            D = gram_matrix.sum(-1)
            D = D.rsqrt()
            D[D == float('inf')] = 0
            gram_matrix = D.view(-1, 1) * gram_matrix * D.view(1, -1)
            print(gram_matrix[:10, :10])
        elif 'nn' in neuralef_loss_cfg['kernel']: #--kernel 16nn_sigm20 --input_l2_normalize, --kernel 16nn_sigm100 --no-input_l2_normalize
            knn = int(neuralef_loss_cfg['kernel'].split("_")[0].replace("nn", ""))
            # the rbf kernel
            A = (FF ** 2).sum(-1).view(-1, 1) + (FF ** 2).sum(-1).view(1, -1) - 2 * FF @ FF.T
            sigma2 = float(neuralef_loss_cfg['kernel'].split("_")[1].replace("sigm", ""))
            A = (- A / 2 / sigma2).exp()
            ret = torch.topk(A, knn, dim=1)
            res = torch.zeros_like(A)
            res.scatter_(1, ret.indices, ret.values)
            gram_matrix = (res + res.T) / 2
            gram_matrix.diagonal().zero_()
            print(gram_matrix[:10, :10])
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
