import torch
import math

from utils.logger import MetricLogger
from metrics import gather_data, compute_metrics
from model import utils
from data.utils import IGNORE_LABEL
import utils.torch as ptu
import numpy as np
import tqdm


def train_one_epoch(
    num_epochs,
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    model.backbone.eval()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    total_num_updates = num_epochs * len(data_loader)
    seg_pred_stats = 0
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            seg_pred, neuralef_loss, neuralef_reg = model.forward(im, return_neuralef_loss=True)
        
        with torch.no_grad():
            loss = criterion(seg_pred, seg_gt)
            seg_pred = seg_pred.permute(0, 2, 3, 1)
            seg_pred_stats += torch.nn.functional.one_hot(seg_pred.argmax(-1), seg_pred.shape[-1]).sum((0, 1, 2)).data.cpu().numpy()

        loss_value = neuralef_loss.item() + neuralef_reg.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                neuralef_loss + neuralef_reg,
                optimizer,
                parameters=model.psi.parameters(),
            )
        else:
            (neuralef_loss + neuralef_reg).backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step(float(num_updates)/total_num_updates)

        torch.cuda.synchronize()

        logger.update(
            neuralef_loss=neuralef_loss.item(),
            neuralef_reg=neuralef_reg.item(),
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
    print(seg_pred_stats)
    

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 400

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
    # find the mapping from ground-truth labels to clustering assignments
    maps = optimal_map(val_seg_gt, val_seg_pred, data_loader.unwrapped.n_cls, IGNORE_LABEL)
    # rotate the clustering assignments
    keys = val_seg_pred.keys()
    for k in keys:
        val_seg_pred[k] = maps[val_seg_pred[k]]
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


def optimal_map(val_seg_gt, val_seg_pred, n_cls, ignore_indice):
    assert ignore_indice >= n_cls, "this code only works for ignore_indices >= n_cls; if not, revise the code accordingly"
    # stats of i-th ground-truth laebl to j-th clustering assignment
    v1, v2 = [], []
    for (k2, v2_) in val_seg_pred.items():
        v1.append(val_seg_gt[k2].reshape((-1)))
        v2.append(v2_.reshape((-1)))
    v1, v2 = np.concatenate(v1), np.concatenate(v2)

    counts = np.zeros((n_cls, n_cls))
    num_pixels = len(v1) - (v1 == ignore_indice).sum()
    for i in tqdm.tqdm(range(n_cls)):
        if i == ignore_indice:
            continue
        pred_i = (v2 == i)
        gt_ = v1[pred_i]
        vv, cc = np.unique(gt_, return_counts=True)
        results = np.where(vv == ignore_indice)[0]
        if len(results) > 0:
            vv = np.delete(vv, list(results))
            cc = np.delete(cc, list(results))
        counts[i, vv] += cc
        # for j in range(n_cls):
        #     if j == ignore_indice:
        #         continue
        #     gt_j = (v1 == j)
        #     counts[i,j] += (pred_i * gt_j).sum()
    assert counts.sum() == num_pixels, (counts.sum(), num_pixels)
    maps = np.argmax(counts, -1)
    return maps

