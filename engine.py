import torch
import math

from utils.logger import MetricLogger
from metrics import gather_data, compute_metrics
from model import utils
from data.utils import IGNORE_LABEL
import utils.torch as ptu
import numpy as np
import tqdm, os
from data.utils import dataset_cat_description, seg_to_rgb, STATS
from data.ade20k import ADE20K_CATS_PATH
from data.pascal_context import PASCAL_CONTEXT_CATS_PATH
from data.cityscapes import CITYSCAPES_CATS_PATH
from PIL import Image
# import denseCRF
import eval_utils
import joblib
from timm import optim
from optim.scheduler import PolynomialLR
from fast_pytorch_kmeans import KMeans

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import torchvision

from reco.utils.utils import get_dataset
from reco.metrics.running_score import RunningScore
from reco.datasets.coco_stuff import coco_stuff_171_to_27


def train_one_epoch(
    num_epochs,
    dataset,
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    tau_max,
    tau_min,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100 if len(data_loader) >= 100 else 50

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

        tau = (math.cos(num_updates / float(num_epochs * len(data_loader)) * math.pi) + 1.) / 2. * (tau_max - tau_min) + tau_min
        tau = None if tau < 0 else tau
        with amp_autocast():
            seg_pred, neuralef_loss, neuralef_reg = model.forward(im, tau=tau, return_neuralef_loss=True, none_mask=(dataset == 'imagenet'))
            if seg_pred is not None:
                loss = criterion(seg_pred, seg_gt)
                with torch.no_grad():
                    mask_ = (seg_gt != IGNORE_LABEL).float()
                    mask_sum_ = mask_.sum()
                    acc = ((seg_pred.argmax(1) == seg_gt).float() * mask_).sum() / mask_sum_
            else:
                loss = torch.zeros_like(neuralef_loss)
                acc = torch.zeros_like(neuralef_loss)

        loss_value = loss.item() + neuralef_loss.item() + neuralef_reg.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, {}, {} stopping training".format(loss.item(), neuralef_loss.item(), neuralef_reg.item()), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss + neuralef_loss + neuralef_reg,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            (loss + neuralef_loss + neuralef_reg).backward()
            optimizer.step()

        num_updates += 1
        if isinstance(lr_scheduler, PolynomialLR):
            lr_scheduler.step_update(num_updates=num_updates)
        else:
            lr_scheduler.step(float(num_updates)/len(data_loader))

        torch.cuda.synchronize()
        logger.update(
            loss=loss.item(),
            acc=acc.item(),
            neuralef_loss=neuralef_loss.item(),
            neuralef_reg=neuralef_reg.item(),
            tau=tau or 0,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
    return logger

@torch.no_grad()
def perform_kmeans(
    model,
    data_loader,
    simulate_one_epoch=False,
):
    n_cls_train = model.psi.psi_dim
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
            kmeans = KMeans(n_clusters=n_cls_train, mode='euclidean', verbose=1)
            assignments = kmeans.fit_predict(cached_features)
            break
        except Exception as e:
            cached_features = cached_features[:int(len(cached_features)*0.8)]

    del cached_features
    cached_features = None

    model.register_buffer("cluster_centers", kmeans.centroids) 
    onehot_assignments = torch.nn.functional.one_hot(assignments, n_cls_train)
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

@torch.no_grad()
def evaluate(
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
    n_cls_train = model.psi.psi_dim
    n_cls = data_loader.unwrapped.n_cls
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 100

    if dataset == 'ade20k':
        _, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
        val_img_folder =  os.path.expandvars("$DATASET/") + 'ade20k/ADEChallengeData2016/images/validation/'
    elif dataset == 'pascal_context':
        _, cat_colors = dataset_cat_description(PASCAL_CONTEXT_CATS_PATH)
        val_img_folder = os.path.expandvars("$DATASET/") + 'pcontext/VOCdevkit/VOC2010/JPEGImages/'
    elif dataset == 'cityscapes':
        _, cat_colors = dataset_cat_description(CITYSCAPES_CATS_PATH)
        val_img_folder =  os.path.expandvars("$DATASET/") + 'cityscapes/leftImg8bit/val/'
    else:
        assert 0

    val_seg_pred = {}
    model.eval()
    model_without_ddp.eval()
    to_perform_crf = []
    counts = 0
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]
        counts += 1

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                n_cls_train,
                batch_size=1,
            )
            
            # val_seg_pred[filename] = seg_pred.argmax(0).cpu().numpy()
            to_perform_crf.append((filename, seg_pred.cpu().numpy()))
            if len(to_perform_crf) == (8 if dataset == 'cityscapes' else 64) or counts == len(data_loader):
                # CRF in multi-process
                results = joblib.Parallel(n_jobs=8, backend='multiprocessing', verbose=0)(
                    [joblib.delayed(process)(to_perform_crf[i][0], to_perform_crf[i][1], val_img_folder) 
                        for i in range(len(to_perform_crf))]
                )
                for i in range(len(to_perform_crf)):
                    val_seg_pred[to_perform_crf[i][0]] = results[i]

                to_perform_crf = []

    val_seg_pred = gather_data(val_seg_pred)
    keys = val_seg_pred.keys()

    # find the mapping from ground-truth labels to clustering assignments
    maps = optimal_map(val_seg_gt, val_seg_pred, n_cls_train, n_cls, IGNORE_LABEL, iou=True)
    print(torch.nn.functional.one_hot(torch.from_numpy(maps).long(), n_cls).sum(0))
    un_matched = (torch.nn.functional.one_hot(torch.from_numpy(maps).long(), n_cls).sum(0) == 0).nonzero()
    print("un_matched", un_matched.numpy())
    # rotate the clustering assignments
    for k in keys:
        val_seg_pred[k] = maps[val_seg_pred[k]]
    
    # the following visualization code works
    vis_dir = log_dir / (str(epoch) + "_" + model.mode)
    vis_dir.mkdir(parents=True, exist_ok=True)
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

def process(k, val_seg_pred_, val_img_folder):
    unary_potentials = val_seg_pred_
    if val_img_folder is None:
        image = k.astype(np.uint8)
    else:
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

def optimal_map(val_seg_gt, val_seg_pred, n_cls_train, n_cls, ignore_indice, iou=False):
   
    # stats of i-th ground-truth laebl to j-th clustering assignment
    if isinstance(val_seg_gt, dict):
        v1, v2 = [], []
        for (k2, v2_) in val_seg_pred.items():
            v1.append(val_seg_gt[k2].reshape((-1)))
            v2.append(v2_.reshape((-1)))
        v1, v2 = np.concatenate(v1), np.concatenate(v2)
    else:
        v1 = val_seg_gt
        v2 = val_seg_pred

    counts = np.zeros((n_cls_train, n_cls))
    count_ignore_indice = np.zeros((n_cls_train,))
    for i in tqdm.tqdm(range(n_cls_train)):
        pred_i = (v2 == i)
        gt_ = v1[pred_i]
        vv, cc = np.unique(gt_, return_counts=True)

        results = np.where(vv == ignore_indice)[0]
        if len(results) > 0:
            assert len(results) == 1
            count_ignore_indice[i] = cc[results[0]]
            vv = np.delete(vv, list(results))
            cc = np.delete(cc, list(results))
        counts[i, vv] = cc
    assert counts.sum() + count_ignore_indice.sum() == len(v1), (counts.sum(), count_ignore_indice.sum(), len(v1))
    if iou:
        counts = counts / (counts.sum(0, keepdims=True) + counts.sum(1, keepdims=True) - counts).clip(1e-8)
    maps = np.argmax(counts, -1)
    return maps


@torch.no_grad()
def reco_protocal_eval(model, dataset_name, split, normalization, dir_ckpt, batch_size=32, n_workers=4):
    vis_dir = dir_ckpt / "reco" / model.mode / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    n_cls_train = model.psi.psi_dim
    if dataset_name == 'pascal_context':
        dir_dataset = os.path.expandvars("$DATASET/") + 'pcontext/VOCdevkit/VOC2010/'
    elif dataset_name == 'cityscapes':
        dir_dataset = os.path.expandvars("$DATASET/") + 'cityscapes'
    dataset, categories, palette = get_dataset(
        dir_dataset=dir_dataset,
        dataset_name=dataset_name,
        split=split,
    )

    running_score = RunningScore(n_classes=dataset.n_categories)
    running_score_crf = RunningScore(n_classes=dataset.n_categories)

    device: torch.device = torch.device("cuda:0")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1 if dataset_name == "kitti_step" else batch_size,
        num_workers=n_workers,
        pin_memory=True
    )

    iter_dataloader = iter(dataloader)
    val_imgs, val_gts, dt_argmaxs, dt_crf_argmaxs = [], [], [], []
    for num_batch in tqdm.tqdm(range(len(dataloader))):
        dict_data = next(iter_dataloader)

        val_img: torch.Tensor = dict_data["img"]  # b x 3 x H x W
        val_gt: torch.LongTensor = dict_data["gt"]  # b x H x W
        raw_val_img = val_img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        val_img = (raw_val_img - torch.tensor(list(STATS[normalization]["mean"]))[:, None, None]) / torch.tensor(list(STATS[normalization]["std"]))[:, None, None]
        dt: torch.Tensor = model(val_img.to(device))  # b x n_cats x H x W
        dt_argmax: torch.Tensor = torch.argmax(dt, dim=1)  # b x H x W

        # CRF in multi-process
        results = joblib.Parallel(n_jobs=16, backend='multiprocessing', verbose=0)( # 16
            [joblib.delayed(process)(raw_val_img[i].permute(1, 2, 0).cpu().numpy(), dt[i].softmax(dim=0).cpu().numpy(), None) 
                for i in range(len(dt))]
        )
        dt_crf_argmax = torch.from_numpy(np.stack(results))
        if num_batch <= 10:
            val_imgs.append(raw_val_img.cpu())
        val_gts.append(val_gt.cpu())
        dt_argmaxs.append(dt_argmax.cpu())
        dt_crf_argmaxs.append(dt_crf_argmax)

    n_cls = dataset.n_categories
    print("n_cls", n_cls, "min_label", min([item.min().item() for item in val_gts]))
    # find the mapping from ground-truth labels to clustering assignments
    maps = torch.from_numpy(optimal_map(torch.cat(val_gts).flatten().numpy(), torch.cat(dt_crf_argmaxs).flatten().numpy(), n_cls_train, n_cls, -1, iou=True)).long()
    print(torch.nn.functional.one_hot(maps, n_cls).sum(0))
    un_matched = (torch.nn.functional.one_hot(maps, n_cls).sum(0) == 0).nonzero()
    print("un_matched", un_matched.numpy())
    
    pbar = tqdm.tqdm(enumerate(zip(val_imgs, val_gts, dt_argmaxs, dt_crf_argmaxs)))
    for num_batch, (val_img, val_gt, dt_argmax, dt_crf_argmax) in pbar:
        # rotate the clustering assignments
        dt_argmax = maps[dt_argmax]
        dt_crf_argmax = maps[dt_crf_argmax]
        if dataset_name == "coco_stuff":  # and not args.coarse_labels:
            dt_coarse: torch.Tensor = torch.zeros_like(dt_argmax)
            dt_coarse_crf: torch.Tensor = torch.zeros_like(dt_crf_argmax)

            for fine, coarse in coco_stuff_171_to_27.items():
                dt_coarse[dt_argmax == fine] = coarse
                dt_coarse_crf[dt_crf_argmax == fine] = coarse
            dt_argmax = dt_coarse
            dt_crf_argmax = dt_coarse_crf

        dt_argmax: np.ndarray = dt_argmax.cpu().numpy()  # b x H x W
        dt_crf_argmax: np.ndarray = dt_crf_argmax.cpu().numpy()  # b x H x W

        running_score.update(label_trues=val_gt.cpu().numpy(), label_preds=dt_argmax)
        running_score_crf.update(label_trues=val_gt.cpu().numpy(), label_preds=dt_crf_argmax)

        miou_crf = running_score_crf.get_scores()[0]["Mean IoU"]
        acc_crf = running_score_crf.get_scores()[0]["Pixel Acc"]

        miou = running_score.get_scores()[0]["Mean IoU"]
        acc = running_score.get_scores()[0]["Pixel Acc"]

        pbar.set_description(
            f"mIoU (bi) {miou:.3f} ({miou_crf:.3f}) | "
            f"Pixel acc (bi) {acc:.3f} ({acc_crf:.3f})"
        )

        if num_batch <= 10:
            pil_img = val_img[0].cpu().numpy()
            pil_img = pil_img * 255.0
            pil_img = np.clip(pil_img, 0, 255)
            val_pil_img: Image.Image = Image.fromarray(pil_img.astype(np.uint8).transpose(1, 2, 0))

            val_gt: np.ndarray = val_gt[0].clone().squeeze(dim=0).cpu().numpy()
            h, w = dt_argmax.shape[-2:]
            unique_labels_dt = np.unique(dt_argmax[0])
            unique_labels_dt_crf = np.unique(dt_crf_argmax[0])
            unique_labels_gt = np.unique(val_gt)

            coloured_dt = np.zeros((h, w, 3), dtype=np.uint8)
            coloured_dt_bi = np.zeros((h, w, 3), dtype=np.uint8)
            coloured_gt = np.zeros((h, w, 3), dtype=np.uint8)
            for ul in unique_labels_dt:
                if ul == -1:
                    continue
                coloured_dt[dt_argmax[0] == ul] = palette[ul]

            for ul in unique_labels_dt_crf:
                if ul == -1:
                    continue
                coloured_dt_bi[dt_crf_argmax[0] == ul] = palette[ul]

            for ul in unique_labels_gt:
                if ul == -1:
                    continue
                coloured_gt[val_gt == ul] = palette[ul]

            nrows, ncols = 1, 4
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 3, nrows * 3))
            for i in range(nrows):
                for j in range(ncols):
                    if j == 0:
                        ax[i, j].imshow(val_pil_img)
                    elif j == 1:
                        ax[i, j].imshow(coloured_gt)
                    elif j == 2:
                        ax[i, j].imshow(coloured_dt)
                    elif j == 3:
                        ax[i, j].imshow(coloured_dt_bi)
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
            plt.tight_layout(pad=0.5)
            plt.savefig(vis_dir/ f"{num_batch:04d}.png")
            plt.close()

    results = running_score.get_scores()[0]
    results.update(running_score.get_scores()[1])

    results_crf = running_score_crf.get_scores()[0]
    results_crf.update(running_score_crf.get_scores()[1])

    json.dump(results, open(vis_dir / f"results.json", "w"))
    json.dump(results_crf, open(vis_dir / f"results_crf.json", "w"))


def vis_eigenfuncs(model, data_loader, amp_autocast, log_dir):
    model.eval()
    data_loader.set_epoch(-1)

    batch = next(iter(data_loader))
    im = batch["im"].to(ptu.device)
    seg_gt = batch["segmentation"].long().to(ptu.device)
    with torch.no_grad():
        with amp_autocast():
            seg_pred = model.forward(im)
    print(seg_pred.shape)

    vis_dir = log_dir / "eigenfuncs"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i in range(seg_pred.shape[0]):
        # images = list(seg_pred[i].data.cpu().numpy())
        # for j, image in enumerate(images[:5]):
        #     print(i, j, image.max(), image.min(), image.shape)
        efs = seg_pred[i][:5].unsqueeze(1).repeat(1, 3, 1, 1)
        # efs = seg_pred[i].softmax(0)[:5].unsqueeze(1).repeat(1, 3, 1, 1)
        efs = (efs - efs.amin(dim=(1,2,3), keepdim=True))/(efs.amax(dim=(1,2,3), keepdim=True) - efs.amin(dim=(1,2,3), keepdim=True))
        
        ims = torch.cat([im[i].unsqueeze(0).add(1.).div(2.), efs], 0)
        # torchvision.utils.save_image(, log_dir / ("eigenfuncs/" + "original_" + str(i) + ".png"))
        torchvision.utils.save_image(ims, log_dir / ("eigenfuncs/" + str(i) + ".png"))