import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from einops import rearrange
from einops_exts import rearrange_many, repeat_many

from model.vit import PatchEmbedding

from utils.torch import freeze_all_layers_
from sklearn.cluster import SpectralClustering
from fast_pytorch_kmeans import KMeans
import numpy as np
import scipy
from sklearn.feature_extraction.image import img_to_graph
from sklearn.neighbors import kneighbors_graph

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

        # self.patch_embed = PatchEmbedding(
        #     self.backbone.patch_embed.image_size,
        #     self.backbone.patch_embed.patch_size,
        #     self.backbone.patch_embed.proj.out_channels,
        #     self.backbone.patch_embed.proj.in_channels
        # )
        # self.patch_embed.proj.weight.data.copy_(self.backbone.patch_embed.proj.weight.data)
        # self.patch_embed.proj.bias.data.copy_(self.backbone.patch_embed.proj.bias.data)

        self.kmeans_cfg = kmeans_cfg
        self.kmeans_n_cls = kmeans_cfg['n_cls']
        self.neuralef_loss_cfg = neuralef_loss_cfg
        self.is_baseline = is_baseline

        # unused buffers
        self.register_buffer('num_calls', torch.Tensor([0]))
        # self.register_buffer("feature_mean", torch.zeros(self.backbone.d_model))

        # the clustering buffers
        self.register_buffer("num_per_cluster", torch.zeros(self.kmeans_n_cls))
        self.register_buffer("cluster_centers", torch.randn(self.kmeans_n_cls, self.psi.num_features+self.backbone.d_model)) # if self.kmeans_cfg['use_hidden_outputs'] else self.psi.out_features))
        
        # a onlien head to check the quality of eigenmaps
        self.online_head = nn.Linear(self.psi.num_features+self.backbone.d_model, n_cls)
        # self.online_head2 = nn.Linear(self.psi.out_features * 3+self.backbone.d_model, n_cls)
        # self.online_head3 = nn.Linear(self.backbone.d_model, n_cls)
        
        # self.cached_FF = []

        self.feat_out = {}
        # self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_affn)
        self.backbone._modules["blocks"][0].register_forward_hook(self.hook_fn_forward_lowlevel)
        self.backbone._modules["blocks"][len(self.backbone._modules["blocks"]) // 2].register_forward_hook(self.hook_fn_forward_midlevel)
        # self.backbone._modules["blocks"][-1].register_forward_hook(self.hook_fn_forward_highlevel)
        
        freeze_all_layers_(self.backbone)
    
    # def hook_fn_forward_affn(self, module, input, output):
    #     output = output.reshape(
    #         output.shape[0], output.shape[1], 3, self.backbone.blocks[0].attn.heads, -1).permute(2, 0, 3, 1, 4)
    #     self.feat_out["affn_feature"] = output[1].transpose(1, 2).flatten(2)[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_lowlevel(self, module, input, output):
        self.feat_out["lowlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    def hook_fn_forward_midlevel(self, module, input, output):
        self.feat_out["midlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]
    
    # def hook_fn_forward_highlevel(self, module, input, output):
    #     self.feat_out["highlevel_feature"] = output[:, 1 + self.backbone.distilled:, :]

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
        Psi = Psi.view(-1, Psi.shape[-1])
        if self.kmeans_cfg['l2_normalize']:
            Psi = F.normalize(Psi, dim=1)

        # Eucelidean distance
        logits = (Psi ** 2).sum(1).view(-1, 1) + (self.cluster_centers ** 2).sum(1).view(1, -1) - 2 * Psi @ self.cluster_centers.T
        logits = -logits
        assignments = logits.argmax(dim=1)
        onehot_assignments = F.one_hot(assignments, self.kmeans_n_cls)
    
        if self.training:
            if isinstance(self.kmeans_cfg['momentum'], float):
                momentum = self.kmeans_cfg['momentum'] * torch.ones(self.kmeans_n_cls, device=Psi.device)
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
        logits = F.one_hot(torch.from_numpy(sc.labels_).to(A.device).long(), self.kmeans_n_cls).float()
        return logits
    
    def forward_features(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        highlevel_feature = self.backbone.forward(im, return_features=True)[:, 1 + self.backbone.distilled:, :]
        # affn_feature = None #self.feat_out["affn_feature"]
        lowlevel_feature = self.feat_out["lowlevel_feature"]
        midlevel_feature = self.feat_out["midlevel_feature"]
        # highlevel_feature = self.feat_out["highlevel_feature"]
        return lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W

    def forward(self, im, seg_gt=None, return_neuralef_loss=False, return_features=False):
        bs = im.shape[0]

        with torch.no_grad():
            lowlevel_feature, midlevel_feature, highlevel_feature, im, H_ori, W_ori, H, W = self.forward_features(im)
            
            # affn_feature = rearrange(affn_feature, "b (h w) c -> b c h w", h=H // self.patch_size)
            # if self.neuralef_loss_cfg['upsample_factor'] > 1:
            #     affn_feature = F.interpolate(affn_feature, scale_factor=self.neuralef_loss_cfg['upsample_factor'], mode="bilinear")
            # affn_feature = rearrange(affn_feature, "b c h w -> (b h w) c")
            
            if self.training:
                im_ = F.interpolate(im, size=(H // self.patch_size * self.neuralef_loss_cfg['upsample_factor'], 
                                             W // self.patch_size * self.neuralef_loss_cfg['upsample_factor']), 
                                   mode="bilinear")
                
                # im = None

                # if seg_gt is not None:
                #     tar_ = seg_gt.clone().detach()
                #     tar_[tar_ >= self.n_cls] = self.n_cls
                #     tar_ = F.one_hot(tar_, self.n_cls + 1).permute(0, 3, 1, 2).half()
                #     tar_ = F.avg_pool2d(tar_, tar_.shape[-1] // im.shape[-1])
                #     tar_ = tar_.permute(0, 2, 3, 1).flatten(0, 2)
                # else:
                #     tar_ = None
            else:
                im_ = None

            # if self.neuralef_loss_cfg['kernel'].split("_")[6] == "lowmidhigh":
            features = torch.cat([lowlevel_feature, midlevel_feature, highlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "low":
            #     features = torch.cat([lowlevel_feature, lowlevel_feature, lowlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "mid":
            #     features = torch.cat([midlevel_feature, midlevel_feature, midlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "high":
            #     features = torch.cat([highlevel_feature, highlevel_feature, highlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "midhigh":
            #     features = torch.cat([midlevel_feature, highlevel_feature, highlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "lowhigh":
            #     features = torch.cat([lowlevel_feature, highlevel_feature, highlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "rawhigh":
            #     raw_feature = rearrange(im, "b c (h p) (w q) -> b (h w) (c p q)", p = self.patch_size, q=self.patch_size)
            #     features = torch.cat([raw_feature, highlevel_feature], dim=-1)
            # elif self.neuralef_loss_cfg['kernel'].split("_")[6] == "rawmid":
            #     raw_feature = rearrange(im, "b c (h p) (w q) -> b (h w) (c p q)", p = self.patch_size, q=self.patch_size)
            #     features = torch.cat([raw_feature, midlevel_feature], dim=-1)

        hidden, Psi = self.psi(features)
        hidden = torch.cat([hidden.clone().detach(), highlevel_feature], -1)
        if return_features:
            # if self.kmeans_cfg['use_hidden_outputs']:
                return hidden
            # return Psi

        if self.is_baseline:
            masks_clustering = self.baseline_clustering(highlevel_feature).view(bs, -1, self.kmeans_n_cls)
        else:
            masks_clustering = None if self.training else self.clustering(hidden).view(bs, -1, self.kmeans_n_cls) # if self.kmeans_cfg['use_hidden_outputs'] else Psi
        masks_clf = self.online_head(hidden).view(bs, -1, self.n_cls)
        # masks_clf2 = self.online_head2(torch.cat([Psi.clone().detach(), final_feature], -1)).view(bs, -1, self.n_cls)
        # masks_clf3 = self.online_head3(highlevel_feature).view(bs, -1, self.n_cls)
        
        masks = rearrange(masks_clf if self.training else masks_clustering, "b (h w) c -> b c h w", h=int(math.sqrt(masks_clf.shape[1])))
        masks = torch.cat([F.interpolate(masks[i*16:min((i+1)*16, len(masks))], size=(H, W), mode="bilinear")
            for i in range(int(math.ceil(float(len(masks))/16.)))])
        masks = unpadding(masks, (H_ori, W_ori))

        # masks2 = rearrange(masks_clf2 if self.training else masks_clustering, "b (h w) c -> b c h w", h=int(math.sqrt(masks_clf2.shape[1])))
        # masks2 = torch.cat([F.interpolate(masks2[i*16:min((i+1)*16, len(masks2))], size=(H, W), mode="bilinear")
        #     for i in range(int(math.ceil(float(len(masks2))/16.)))])
        # masks2 = unpadding(masks2, (H_ori, W_ori))

        # masks3 = rearrange(masks_clf3 if self.training else masks_clustering, "b (h w) c -> b c h w", h=int(math.sqrt(masks_clf3.shape[1])))
        # masks3 = torch.cat([F.interpolate(masks3[i*16:min((i+1)*16, len(masks3))], size=(H, W), mode="bilinear")
        #     for i in range(int(math.ceil(float(len(masks3))/16.)))])
        # masks3 = unpadding(masks3, (H_ori, W_ori))

        if return_neuralef_loss:
            if self.psi.num_layers == 0:
                neuralef_loss = torch.tensor([0.]).to(im.device)
                neuralef_reg = torch.tensor([0.]).to(im.device)
            else:
                neuralef_loss, neuralef_reg = cal_neuralef_loss(lowlevel_feature, midlevel_feature, highlevel_feature, Psi, im_, self.neuralef_loss_cfg)
                # if len(self.cached_FF) < self.neuralef_loss_cfg["cache_size"]:
                #     with torch.no_grad():
                #         self.cached_FF.append(F.normalize(FF, dim=1) if self.neuralef_loss_cfg['input_l2_normalize'] == True else FF)
                # if len(self.cached_FF) == self.neuralef_loss_cfg["cache_size"]:
                #     del self.cached_FF[0]
                # with torch.no_grad():
                #     self.cached_FF.append(F.normalize(FF, dim=1) if self.neuralef_loss_cfg['input_l2_normalize'] == True else FF)
            return masks, neuralef_loss, neuralef_reg #, masks2, masks3
       
        return masks


def cal_neuralef_loss(lowlevel_feature, midlevel_feature, highlevel_feature, Psi, im, neuralef_loss_cfg, tar=None):
    if Psi.dim() == 3:
        Psi = Psi.flatten(0, 1)
    Psi *= math.sqrt(neuralef_loss_cfg['t'] / Psi.shape[0])
    with torch.no_grad():
        # if neuralef_loss_cfg['input_l2_normalize'] == True:
        #     FF = F.normalize(FF, dim=1)

        # if tar is not None:
        #     A = tar @ tar.T
        #     A.fill_diagonal_(0.)

        #     D = A.sum(-1)#.mean()
        #     D = D.rsqrt()
        #     gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1)).type_as(Psi)
        #     # gram_matrix = A.type_as(Psi)

        # elif 'normalized_adjacency' in neuralef_loss_cfg['kernel']:
        #     # the affinity
        #     A = (FF @ FF.T)
        #     A.fill_diagonal_(0.)
        #     if 'thresholded' in neuralef_loss_cfg['kernel']:
        #         A.clamp_(min=0.)
            
        #     # combine with pixelwise affinity
        #     if neuralef_loss_cfg['pixelwise_adj_weight'] > 0:
        #         im = im.add_(1.).div_(2.)
        #         bs, h, w = im.shape[0], im.shape[2], im.shape[3]
        #         x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, -1, 1).repeat(bs, 1, 1)
        #         y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, -1, 1).repeat(bs, 1, 1)
        #         im = im.flatten(2).permute(0, 2, 1) / neuralef_loss_cfg['pixelwise_adj_div_factor']
        #         A_p = None
        #         for k, distance_weight in zip([20, 10], [2.0, 0.1]):
        #             im2 = torch.cat([im, 
        #                             x_.mul(distance_weight),
        #                             y_.mul(distance_weight)], -1)
        #             with torch.cuda.amp.autocast():
        #                 euc_dist = -torch.cdist(im2, im2)
        #                 euc_dist.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        #                 ret = torch.topk(euc_dist, k, dim=2)
        #             res = torch.zeros(*euc_dist.shape, device=euc_dist.device, dtype=bool)
        #             res.scatter_(2, ret.indices, torch.ones_like(ret.values).bool())
        #             if A_p is None:
        #                 A_p = torch.logical_and(res, res.permute(0, 2, 1))
        #             else:
        #                 A_p = torch.logical_or(A_p, torch.logical_and(res, res.permute(0, 2, 1)))

        #         for i in range(bs):
        #             A[i*(h*w):(i+1)*(h*w), i*(h*w):(i+1)*(h*w)] += neuralef_loss_cfg['pixelwise_adj_weight'] * A_p[i].float()
                
        #         # # im_np = im.permute(0, 2, 3, 1).data.cpu().numpy()
        #         # x_ = np.tile(np.linspace(0, 1, w), h).reshape((-1, 1))
        #         # y_ = np.repeat(np.linspace(0, 1, h), w).reshape((-1, 1))
        #         # xy = np.concatenate([x_, y_], 1)
        #         # im_np = im.permute(0, 2, 3, 1).flatten(1, 2).data.cpu().numpy() / neuralef_loss_cfg['pixelwise_adj_div_factor']
        #         # for i in range(bs):
        #         #     A_p_i = 0
        #         #     for k, distance_weight in zip([20, 10], [2.0, 0.1]):
        #         #         connectivity = kneighbors_graph(
        #         #             np.concatenate([im_np[i], xy * distance_weight], 1), n_neighbors=k, include_self=False, n_jobs=-1
        #         #         )
        #         #         A_p_i = A_p_i + connectivity + connectivity.T #connectivity.multiply(connectivity.T)
                    
        #         #     # A_p_i = img_to_graph(im_np[i])
                    
        #         #     A_p_i = A_p_i.tocoo()
        #         #     values = A_p_i.data
        #         #     indices = np.vstack((A_p_i.row, A_p_i.col))
        #         #     A_p_i = torch.sparse_coo_tensor(torch.LongTensor(indices).to(A.device), torch.FloatTensor(values).to(A.device).fill_(1.), torch.Size(A_p_i.shape)).to_dense()
        #         #     # print(A_p_i[:10, :10])
        #         #     A[i*(h*w):(i+1)*(h*w), i*(h*w):(i+1)*(h*w)] += neuralef_loss_cfg['pixelwise_adj_weight'] * A_p_i
            
        #     # estimate D
        #     if len(cached_FF) == 0:
        #         D = A.sum(-1) # / (A.shape[-1] - 1)
        #     else:
        #         b_ = cached_FF[0].shape[0] // 8
        #         assert cached_FF[0].shape[0] % 8 == 0
        #         D = sum([(FF @ cached_FF_[i*b_:(i+1)*b_].T).clamp(min=0).sum(-1) for cached_FF_ in cached_FF for i in range(8)]) + A.sum(-1)
        #         D /= sum([cached_FF_.shape[0] for cached_FF_ in cached_FF]) + A.shape[-1] - 1
        #     D = D.rsqrt()
        #     # D = torch.tensor([1.]).to(A.device)

        #     # the gram matrix
        #     gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1))
        #     # gram_matrix = gram_matrix.float()
        #     # gram_matrix.diagonal().add_(1.)
        if 'nearestn' in neuralef_loss_cfg['kernel']:
            num_ = int(neuralef_loss_cfg['kernel'].split("_")[0].replace("nearestn", ""))
            dist_measure = neuralef_loss_cfg['kernel'].split("_")[1]
            weight_high = float(neuralef_loss_cfg['kernel'].split("_")[2])
            weight_mid = float(neuralef_loss_cfg['kernel'].split("_")[3])
            num_2 = int(neuralef_loss_cfg['kernel'].split("_")[4])
            num_3 = int(neuralef_loss_cfg['kernel'].split("_")[5])

            with torch.cuda.amp.autocast():
                if dist_measure == "cosine":
                    highlevel_feature_ = F.normalize(highlevel_feature.flatten(0, 1), dim=-1) #
                    A = highlevel_feature_ @ highlevel_feature_.transpose(-1, -2)
                    A.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                    A.clamp_(min=0.)
                    
                    ret = torch.topk(A, num_, dim=-1)
                    res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
                    res.scatter_(-1, ret.indices, ret.values)
                    res.add_(res.T).div_(2.)
                    A = res # * weight_high

                    # ret = torch.topk(A, num_, dim=2)
                    # res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
                    # res.scatter_(2, ret.indices, ret.values)
                    # res.add_(res.permute(0, 2, 1)).div_(2.)
                    # A = torch.block_diag(*res)

                    '''
                    midlevel_feature_ = F.normalize(midlevel_feature.flatten(0, 1), dim=-1) #.flatten(0, 1)
                    A2 = midlevel_feature_ @ midlevel_feature_.transpose(-1, -2)
                    A2.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                    A2.clamp_(min=0.)
                    
                    ret = torch.topk(A2, num_, dim=-1)
                    res = torch.zeros(*A2.shape, device=A2.device, dtype=A2.dtype)
                    res.scatter_(-1, ret.indices, ret.values)
                    res.add_(res.T).div_(2.)
                    A2 = res # * weight_mid

                    # ret = torch.topk(A_, num_, dim=2)
                    # res = torch.zeros(*A_.shape, device=A_.device, dtype=A_.dtype)
                    # res.scatter_(2, ret.indices, ret.values)
                    # res.add_(res.permute(0, 2, 1)).div_(2.)
                    # A += torch.block_diag(*res) * weight_mid
                    
                    lowlevel_feature_ = F.normalize(lowlevel_feature.flatten(0, 1), dim=-1) #.flatten(0, 1)
                    A3 = lowlevel_feature_ @ lowlevel_feature_.transpose(-1, -2)
                    A3.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                    A3.clamp_(min=0.)
                    
                    ret = torch.topk(A3, num_, dim=-1)
                    res = torch.zeros(*A3.shape, device=A3.device, dtype=A3.dtype)
                    res.scatter_(-1, ret.indices, ret.values)
                    res.add_(res.T).div_(2.)
                    A3 = res #* weight_low
                    
                    # ret = torch.topk(A_, num_, dim=2)
                    # res = torch.zeros(*A_.shape, device=A_.device, dtype=A_.dtype)
                    # res.scatter_(2, ret.indices, ret.values)
                    # res.add_(res.permute(0, 2, 1)).div_(2.)
                    # A += torch.block_diag(*res) * weight_low
                    '''

                    im = im.add_(1.).div_(2.)
                    bs, h, w = im.shape[0], im.shape[2], im.shape[3]
                    x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, -1, 1).repeat(bs, 1, 1)
                    y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, -1, 1).repeat(bs, 1, 1)
                    im = im.flatten(2).permute(0, 2, 1) / neuralef_loss_cfg['pixelwise_adj_div_factor']
                    A_p = None
                    for k, distance_weight in zip([num_2, num_3], [2.0, 0.1]):
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


                # elif dist_measure == 'innerp':
                #     A = FF @ FF.T

                #     ## new code for results of using innerp
                #     A.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                #     ret = torch.topk(A, num_, dim=-1)
                #     res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
                #     res.scatter_(-1, ret.indices, ret.values)
                #     res.add_(res.T).div_(2.)
                #     A = res

                # elif dist_measure == 'l2':
                #     A = -torch.cdist(FF, FF)
                #     # original code for results of using l2
                #     A.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                #     ret = torch.topk(A, num_, dim=-1)
                #     res = torch.zeros(*A.shape, device=A.device, dtype=A.dtype)
                #     res.scatter_(-1, ret.indices, torch.ones_like(ret.values))
                #     res.add_(res.T).div_(2.)
                #     A = res
                
                # if neuralef_loss_cfg['pixelwise_adj_weight'] > 0:
                #     im = im.add_(1.).div_(2.)
                #     bs, h, w = im.shape[0], im.shape[2], im.shape[3]
                #     x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, -1, 1).repeat(bs, 1, 1)
                #     y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, -1, 1).repeat(bs, 1, 1)
                #     im = im.flatten(2).permute(0, 2, 1) / neuralef_loss_cfg['pixelwise_adj_div_factor']
                #     A_p = None
                #     for k, distance_weight in zip([num_2, num_3], [2.0, 0.1]):
                #         if k == 0:
                #             continue
                #         im2 = torch.cat([im, 
                #                         x_.mul(distance_weight),
                #                         y_.mul(distance_weight)], -1)
                #         euc_dist = -torch.cdist(im2, im2)
                #         euc_dist.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
                #         ret = torch.topk(euc_dist, k, dim=2)
                #         res = torch.zeros(*euc_dist.shape, device=euc_dist.device, dtype=bool)
                #         res.scatter_(2, ret.indices, torch.ones_like(ret.values).bool())
                #         if A_p is None:
                #             A_p = torch.logical_or(res, res.permute(0, 2, 1))
                #         else:
                #             A_p = torch.logical_or(A_p, torch.logical_or(res, res.permute(0, 2, 1)))
                #     if A_p is not None:
                #         for i in range(bs):
                #             A[i*(h*w):(i+1)*(h*w), i*(h*w):(i+1)*(h*w)] += neuralef_loss_cfg['pixelwise_adj_weight'] * A_p[i].type_as(A)

                D = A.sum(-1).rsqrt()
                gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1)).type_as(Psi)

                D2 = A2.sum(-1).rsqrt()
                gram_matrix2 = A2.mul_(D2.view(1, -1)).mul_(D2.view(-1, 1)).type_as(Psi)

                # D3 = A3.sum(-1).rsqrt()
                # gram_matrix3 = A3.mul_(D3.view(1, -1)).mul_(D3.view(-1, 1)).type_as(Psi)
        
        # elif 'ball' in neuralef_loss_cfg['kernel']:
        #     max_dist_ = float(neuralef_loss_cfg['kernel'].split("_")[0])
        #     A = (torch.cdist(FF, FF) < max_dist_).float()
        #     A.diagonal(dim1=-2, dim2=-1).fill_(0)
        #     # print(A.mean())
        #     D = A.sum(-1) / (A.shape[-1] - 1)
        #     D = D.rsqrt()
        #     # D[D==float('nan')] = 0
        #     D[D==float('inf')] = 0
        #     # D[D==float('-inf')] = 0
        #     # print(D)
        #     gram_matrix = A.mul_(D.view(1, -1)).mul_(D.view(-1, 1))

        # elif 'normalized_adj_instancewise' in neuralef_loss_cfg['kernel']:
        #     A = (FF.view(im.shape[0], -1, FF.shape[-1]) @ FF.view(im.shape[0], -1, FF.shape[-1]).permute(0, 2, 1))
        #     A.diagonal(dim1=-2, dim2=-1).fill_(0)
        #     A.clamp_(min=0.)
        #     D = A.sum(-1) #/(A.shape[-1] - 1)
        #     D = D.rsqrt()
        #     gram_matrix = A.mul_(D.unsqueeze(2)).mul_(D.unsqueeze(1))
        #     gram_matrix.diagonal(dim1=-2, dim2=-1).add_(1.)
        # elif 'pixel_only' in neuralef_loss_cfg['kernel']:
        #     im = im.add_(1.).div_(2.)
        #     bs, h, w = im.shape[0], im.shape[2], im.shape[3]
        #     x_ = torch.tile(torch.linspace(0, 1, w), (h,)).to(im.device).view(1, -1, 1).repeat(bs, 1, 1)
        #     y_ = torch.repeat_interleave(torch.linspace(0, 1, h).to(im.device), w).view(1, -1, 1).repeat(bs, 1, 1)
        #     im = im.flatten(2).permute(0, 2, 1) / neuralef_loss_cfg['pixelwise_adj_div_factor']
        #     sigma2 = float(neuralef_loss_cfg['kernel'].split("_")[-1].replace("sigma", ""))
        #     A_p = None
        #     for k, distance_weight in zip([20, 10], [2.0, 0.1]):
        #         im2 = torch.cat([im, 
        #                         x_.mul(distance_weight),
        #                         y_.mul(distance_weight)], -1)
        #         with torch.cuda.amp.autocast():
        #             euc_dist = -torch.cdist(im2, im2)
        #             euc_dist.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        #             if 'full' in neuralef_loss_cfg['kernel']:
        #                 res = euc_dist.mul(sigma2).exp().float()
        #             else:
        #                 ret = torch.topk(euc_dist, k, dim=2)
        #                 res = torch.zeros(*euc_dist.shape, device=euc_dist.device)
        #                 res.scatter_(2, ret.indices, ret.values.mul(sigma2).exp())
        #                 res = res.float()
        #         if A_p is None:
        #             A_p = (res + res.permute(0, 2, 1)) / 2.
        #         else:
        #             A_p += (res + res.permute(0, 2, 1)) / 2.
        #     A = A_p.float()
        #     D = A.sum(-1)
        #     D = D.rsqrt()
        #     gram_matrix = A.mul_(D.unsqueeze(2)).mul_(D.unsqueeze(1))
        else:
            assert False

    if neuralef_loss_cfg['asymmetric']:
        with torch.no_grad():
            idx = torch.randperm(Psi.shape[0])
            idx1 = idx[:Psi.shape[0]//2]
            idx2 = idx[Psi.shape[0]//2:]
            Psi1 = Psi[idx1]
            Psi2 = Psi[idx2]
            del Psi
            gram_matrix = gram_matrix[idx1]
            gram_matrix = gram_matrix[:, idx2]
        
        R = Psi1.T @ gram_matrix @ Psi2
        if neuralef_loss_cfg['no_sg']:
            R_hat1 = Psi1.T @ gram_matrix @ Psi2
            R_hat2 = Psi2.T @ gram_matrix.T @ Psi1
        else:
            R_hat1 = Psi1.T.detach() @ gram_matrix @ Psi2
            R_hat2 = Psi2.T.detach() @ gram_matrix.T @ Psi1
        loss = - R.diagonal().sum() / Psi1.shape[1]
        reg = ((R_hat1 ** 2).triu(1).sum() + (R_hat2 ** 2).triu(1).sum())  / Psi1.shape[1]/2
    elif neuralef_loss_cfg['instancewise']:
        Psi1 = Psi2 = Psi
        Psi1 = Psi1.view(im.shape[0], -1, Psi1.shape[-1])
        Psi2 = Psi2.view(im.shape[0], -1, Psi2.shape[-1])
        R = torch.einsum("bmd,bmn,bne->bde", Psi1, gram_matrix, Psi2)
        if neuralef_loss_cfg['no_sg']:
            R_hat = R
        else:
            R_hat = torch.einsum("bmd,bmn,bne->bde", Psi1.detach(), gram_matrix, Psi2)
        loss = - R.diagonal(1, 2).sum(-1).mean() # / Psi1.shape[0]
        reg = (R_hat ** 2).triu(1).sum([1, 2]).mean() # / Psi1.shape[0] # / R.diagonal().detach().abs()
    else:
        Psi1, Psi2 = Psi.chunk(2, dim=-1) #, Psi3
        loss, reg = 0, 0
        for Psi_, gram_matrix_, weight_ in zip([Psi1, Psi2], [gram_matrix, gram_matrix2], [weight_high, weight_mid]):
            R = Psi_.T @ gram_matrix_ @ Psi_
            if neuralef_loss_cfg['no_sg']:
                R_hat = R
            else:
                R_hat = Psi_.T.detach() @ gram_matrix_ @ Psi_
            loss += - R.diagonal().sum() / Psi.shape[1] * weight_
            reg += (R_hat ** 2).triu(1).sum() / Psi.shape[1] * weight_ # / R.diagonal().detach().abs()
    return loss, reg * neuralef_loss_cfg['alpha']
