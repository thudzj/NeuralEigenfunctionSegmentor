import torch
import math

from utils.logger import MetricLogger
from metrics import gather_data, compute_metrics
from model import utils
from data.utils import IGNORE_LABEL
import utils.torch as ptu
import numpy as np
import tqdm, os
from data.utils import dataset_cat_description, seg_to_rgb
from data.ade20k import ADE20K_CATS_PATH
from data.pascal_context import PASCAL_CONTEXT_CATS_PATH
from data.cityscapes import CITYSCAPES_CATS_PATH
from PIL import Image
from fast_pytorch_kmeans import KMeans
# import denseCRF
import eval_utils
import joblib
from timm import optim
from optim.scheduler import PolynomialLR


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    is_linear_probe
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100 if len(data_loader) >= 100 else 50

    if is_linear_probe:
        model.eval()
    else:
        model.train()
        model.backbone.eval()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    batch_size = None
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)
        if batch_size is None:
            batch_size = im.shape[0]
        if im.shape[0] < batch_size:
            break

        with amp_autocast():
            if is_linear_probe:
                seg_pred = model.forward(im)
                neuralef_loss, neuralef_reg = torch.tensor([0.]).to(ptu.device), torch.tensor([0.]).to(ptu.device)
            else:
                seg_pred, neuralef_loss, neuralef_reg = model.forward(im, return_neuralef_loss=True)
            loss = criterion(seg_pred, seg_gt)
            # loss2 = criterion(seg_pred2, seg_gt)

            with torch.no_grad():
                mask_ = (seg_gt != IGNORE_LABEL).float()
                mask_sum_ = mask_.sum()
                acc = ((seg_pred.argmax(1) == seg_gt).float() * mask_).sum() / mask_sum_
                # acc2 = ((seg_pred2.argmax(1) == seg_gt).float() * mask_).sum() / mask_sum_
        
        loss_value = loss.item() + neuralef_loss.item() + neuralef_reg.item() # + loss2.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss + neuralef_loss + neuralef_reg, # + loss2
                optimizer,
                parameters=model.parameters(),
            )
        else:
            (loss + neuralef_loss + neuralef_reg).backward() # + loss2
            optimizer.step()

        num_updates += 1
        if isinstance(lr_scheduler, PolynomialLR):
            lr_scheduler.step_update(num_updates=num_updates)
        else:
            lr_scheduler.step(float(num_updates)/len(data_loader))

        torch.cuda.synchronize()
        logger.update(
            loss=loss.item(),
            # loss2=loss2.item(),
            acc=acc.item(),
            # acc2=acc2.item(),
            neuralef_loss=neuralef_loss.item(),
            neuralef_reg=neuralef_reg.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
    return logger

@torch.no_grad()
def fit_pca(
    model,
    data_loader,
):
    dim = model.kmeans_cfg['dim']
    if dim == 0:
        return 

    model.eval()
    data_loader.set_epoch(997)
    mean, count = 0, 0
    for i, batch in tqdm.tqdm(enumerate(data_loader), 'est feature mean'):
        im = batch["im"].to(ptu.device)
        feature = model.forward(im, return_features=True)
        mean += feature.view(-1, feature.shape[-1]).sum(0)
        count += feature.numel() // feature.shape[-1]        
        torch.cuda.synchronize()
    mean /= count

    cov = 0
    for i, batch in tqdm.tqdm(enumerate(data_loader), 'est feature cov'):
        im = batch["im"].to(ptu.device)
        feature = model.forward(im, return_features=True)
        feature = feature.view(-1, feature.shape[-1]) - mean
        cov += feature.T @ feature
        torch.cuda.synchronize()
    cov /= count

    eig_values, eig_vectors = torch.linalg.eigh(cov)
    proj_matrix = eig_vectors[:, range(-1, -(dim+1), -1)]

    model.register_buffer("feature_mean", mean) 
    model.register_buffer("proj_matrix", proj_matrix) 

@torch.no_grad()
def tune_clusters(
    model,
    data_loader,
    simulate_one_epoch=False,
):
    model.eval()
    data_loader.set_epoch(999)
    cached_features = []
    for i, batch in tqdm.tqdm(enumerate(data_loader), 'init cluster centers'):
        im = batch["im"].to(ptu.device)
        feature = model.forward(im, return_features=True)
        feature = feature.reshape(-1, feature.shape[-1])
        cached_features.append(feature)
        torch.cuda.synchronize()
        if sum([item.numel() for item in cached_features]) > 2000000000:
            break
    cached_features = torch.cat(cached_features)
    
    while 1:
        try:
            kmeans = KMeans(n_clusters=model.kmeans_n_cls, mode='euclidean', verbose=1)
            assignments = kmeans.fit_predict(cached_features)
            break
        except Exception as e:
            cached_features = cached_features[:int(len(cached_features)*0.8)]

    del cached_features
    cached_features = None

    model.register_buffer("cluster_centers", kmeans.centroids) 
    onehot_assignments = torch.nn.functional.one_hot(assignments, model.kmeans_n_cls)
    model.register_buffer("num_per_cluster", onehot_assignments.long().sum(0))
    del kmeans, onehot_assignments

    if simulate_one_epoch:
        data_loader.set_epoch(1000)
        for i, batch in tqdm.tqdm(enumerate(data_loader), 'tune centers'):
            im = batch["im"].to(ptu.device)
            feature = model.forward(im, return_features=True)
            feature = feature.reshape(-1, feature.shape[-1])
            model.clustering(feature, update=True)
            torch.cuda.synchronize()
        data_loader.set_epoch(1001)
        for i, batch in tqdm.tqdm(enumerate(data_loader), 'tune centers'):
            im = batch["im"].to(ptu.device)
            feature = model.forward(im, return_features=True)
            feature = feature.reshape(-1, feature.shape[-1])
            model.clustering(feature, update=True)
            torch.cuda.synchronize()

@torch.no_grad()
def evaluate(
    is_linear_probe,
    dataset,
    epoch,
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
    log_dir,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 400

    if dataset == 'ade20k':
        cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
        val_img_folder =  os.path.expandvars("$DATASET/") + 'ade20k/ADEChallengeData2016/images/validation/'
    elif dataset == 'pascal_context':
        cat_names, cat_colors = dataset_cat_description(PASCAL_CONTEXT_CATS_PATH)
        val_img_folder = os.path.expandvars("$DATASET/") + 'pcontext/VOCdevkit/VOC2010/JPEGImages/'
    elif dataset == 'cityscapes':
        cat_names, cat_colors = dataset_cat_description(CITYSCAPES_CATS_PATH)
        val_img_folder =  os.path.expandvars("$DATASET/") + 'ade20k/ADEChallengeData2016/images/validation/'
    else:
        assert 0

    val_seg_pred = {}
    model.eval()
    model_without_ddp.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                is_linear_probe,
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    keys = val_seg_pred.keys()

    if not is_linear_probe:
        # find the mapping from ground-truth labels to clustering assignments
        maps = optimal_map(val_seg_gt, val_seg_pred, model.kmeans_n_cls, data_loader.unwrapped.n_cls, IGNORE_LABEL, iou=False)
        # rotate the clustering assignments
        for k in keys:
            val_seg_pred[k] = maps[val_seg_pred[k]]
    
    # list_keys = list(keys)
    # # CRF in multi-process
    # results = joblib.Parallel(n_jobs=16, backend='multiprocessing', verbose=9)(
    #     [joblib.delayed(process)(list_keys[i], val_seg_pred[list_keys[i]], model.n_cls, val_img_folder) 
    #           for i in range(len(list_keys))]
    # )
    # for i in range(len(list_keys)):
    #     val_seg_pred[list_keys[i]] = results[i]

    # the following visualization code works only for ade20k
    vis_dir = log_dir / (str(epoch) + model.kmeans_cfg['feature'] + ("_linearprobe" if is_linear_probe else ""))
    vis_dir.mkdir(parents=True, exist_ok=True)
    cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
    for i, k in enumerate(keys):
        seg_rgb = seg_to_rgb(torch.from_numpy(val_seg_pred[k])[None, :, :], cat_colors)
        seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        pil_seg = Image.fromarray(seg_rgb[0])

        pil_im = Image.open(val_img_folder + k).copy()
        dst = Image.new('RGB', (pil_im.width*3, pil_im.height))
        dst.paste(pil_im, (0, 0))
        dst.paste(pil_seg, (pil_im.width, 0))
        pil_blend = Image.blend(pil_im, pil_seg, 0.2).convert("RGB")
        dst.paste(pil_blend, (pil_im.width*2, 0))
        dst.save(vis_dir / "{}.png".format(i))
        if i == 9:
            break

    # estimate performance
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger

def process(k, val_seg_pred_, n_cls, val_img_folder):
    
    unary_potentials = np.transpose(
        np.reshape(
            np.eye(n_cls)[val_seg_pred_.reshape(-1)], 
            (val_seg_pred_.shape[0], val_seg_pred_.shape[1], n_cls)
        ), 
        (2, 0, 1)
    )
    image = np.array(Image.open(val_img_folder + k).convert('RGB')).astype(np.uint8)

    ITER_MAX= 10
    POS_W= 3
    POS_XY_STD= 1
    BI_W= 4
    BI_XY_STD= 67
    BI_RGB_STD= 3
    postprocessor = eval_utils.DenseCRF(
        iter_max=ITER_MAX,
        pos_xy_std=POS_XY_STD,
        pos_w=POS_W,
        bi_xy_std=BI_XY_STD,
        bi_rgb_std=BI_RGB_STD,
        bi_w=BI_W,
    )
    prob = postprocessor(image, unary_potentials)
    return np.argmax(prob, axis=0)

def optimal_map(val_seg_gt, val_seg_pred, kmeans_n_cls, n_cls, ignore_indice, iou=False):
   
    # stats of i-th ground-truth laebl to j-th clustering assignment
    v1, v2 = [], []
    for (k2, v2_) in val_seg_pred.items():
        v1.append(val_seg_gt[k2].reshape((-1)))
        v2.append(v2_.reshape((-1)))
    v1, v2 = np.concatenate(v1), np.concatenate(v2)

    # print('Using majority voting for matching')
    # match = eval_utils.majority_vote(v2, v1, preds_k=kmeans_n_cls, targets_k=n_cls, n_jobs=8)
    # return np.array([aa[1] for aa in match])

    counts = np.zeros((kmeans_n_cls, n_cls))
    if ignore_indice >= n_cls:
        count_ignore_indice = np.zeros((kmeans_n_cls,))
    for i in tqdm.tqdm(range(kmeans_n_cls)):
        pred_i = (v2 == i)
        gt_ = v1[pred_i]
        vv, cc = np.unique(gt_, return_counts=True)
        if ignore_indice >= n_cls:
            results = np.where(vv == ignore_indice)[0]
            if len(results) > 0:
                assert len(results) == 1
                count_ignore_indice[i] = cc[results[0]]
                vv = np.delete(vv, list(results))
                cc = np.delete(cc, list(results))
        counts[i, vv] = cc
    if ignore_indice >= n_cls:
        assert counts.sum() + count_ignore_indice.sum() == len(v1), (counts.sum(), count_ignore_indice.sum(), len(v1))
        # for i in range(maps.shape[0]):
        #     if maps[i] == counts.shape[1]:
        #         maps[i] = ignore_indice
        #         print("cluster", i, "set to ignore_indice", ignore_indice)
    else:
        assert counts.sum() == len(v1), (counts.sum(), len(v1))
    if iou:
        counts = counts / (counts.sum(0, keepdims=True) + counts.sum(1, keepdims=True) - counts).clip(1e-8)
    maps = np.argmax(counts, -1)
    return maps

