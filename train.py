import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed
import utils.torch as ptu
import config

from model.factory import create_segmenter
from optim.factory import create_optimizer, create_scheduler
from data.factory import create_dataset
from model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress

from utils.distributed import sync_model
from engine import train_one_epoch, evaluate, tune_clusters



@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--port", default=12345, type=int)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_small_patch16_384", type=str)
@click.option("--optimizer", default="adamw", type=str)
@click.option("--scheduler", default="cosine", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=0.01, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option("--kernel", default=None, type=str)
@click.option("--input_l2_normalize/--no-input_l2_normalize", default=False, is_flag=True)
@click.option("--alpha", default=None, type=float)
@click.option("--psi_res/--no-psi_res", default=False, is_flag=True)
@click.option("--psi_transformer/--no-psi_transformer", default=False, is_flag=True)
@click.option("--psi_k", default=None, type=int)
@click.option("--psi_num_features", default=None, type=int)
@click.option("--psi_norm_type", default=None, type=str)
@click.option("--psi_act_type", default=None, type=str)
@click.option("--psi_num_layers", default=None, type=int)
@click.option("--psi_projector/--no-psi_projector", default=False, is_flag=True)
@click.option("--t_for_neuralef", default=None, type=float)
@click.option("--pixelwise_adj_weight", default=0, type=float)
@click.option("--pixelwise_adj_div_factor", default=1, type=float)
@click.option("--pixelwise_adj_knn", default=10, type=int)
@click.option("--instancewise/--no-instancewise", default=False, is_flag=True)
@click.option("--asymmetric/--no-asymmetric", default=False, is_flag=True)
@click.option("--no_sg/--no-no_sg", default=False, is_flag=True)
@click.option("--upsample_factor", default=1, type=int)
@click.option("--is_baseline/--no-is_baseline", default=False, is_flag=True)
@click.option("--kmeans_n_cls", default=None, type=int)
@click.option("--kmeans_use_hidden_outputs/--no-kmeans_use_hidden_outputs", default=False, is_flag=True)
@click.option("--kmeans_l2_normalize/--no-kmeans_l2_normalize", default=False, is_flag=True)
@click.option("--cache_size", default=100, type=int)

def main(
    log_dir,
    dataset,
    port,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
    kernel,
    input_l2_normalize,
    alpha,
    psi_res,
    psi_transformer,
    psi_k,
    psi_num_features,
    psi_norm_type,
    psi_act_type,
    psi_num_layers,
    psi_projector,
    t_for_neuralef,
    pixelwise_adj_weight,
    pixelwise_adj_div_factor,
    pixelwise_adj_knn,
    instancewise,
    asymmetric,
    no_sg,
    upsample_factor,
    is_baseline,
    kmeans_n_cls,
    kmeans_use_hidden_outputs,
    kmeans_l2_normalize,
    cache_size
):
    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process(port)

    # set up configuration
    cfg = config.load_config()
    backbone_cfg = cfg["backbone"][backbone]
    psi_cfg = cfg["psi"]
    psi_cfg['res'] = psi_res
    psi_cfg['transformer'] = psi_transformer
    if psi_k is not None:
        psi_cfg['k'] = psi_k
    if psi_num_features is not None:
        psi_cfg['num_features'] = psi_num_features
    if psi_norm_type is not None:
        psi_cfg['norm_type'] = psi_norm_type
    if psi_act_type is not None:
        psi_cfg['act_type'] = psi_act_type
    psi_cfg['projector'] = psi_projector
    if psi_num_layers is not None:
        psi_cfg['num_layers'] = psi_num_layers
    kmeans_cfg = cfg["kmeans"]
    if kmeans_n_cls is not None:
        kmeans_cfg['n_cls'] = kmeans_n_cls
    kmeans_cfg['use_hidden_outputs'] = kmeans_use_hidden_outputs
    kmeans_cfg['l2_normalize'] = kmeans_l2_normalize
    neuralef_loss_cfg = cfg['neuralef']
    if kernel is not None:
        neuralef_loss_cfg['kernel'] = kernel
    if alpha is not None:
        neuralef_loss_cfg['alpha'] = alpha
    neuralef_loss_cfg['input_l2_normalize'] = input_l2_normalize
    if t_for_neuralef is not None:
        neuralef_loss_cfg['t'] = t_for_neuralef
    neuralef_loss_cfg['instancewise'] = instancewise
    neuralef_loss_cfg['asymmetric'] = asymmetric
    neuralef_loss_cfg['no_sg'] = no_sg
    
    neuralef_loss_cfg['upsample_factor'] = upsample_factor
    neuralef_loss_cfg['pixelwise_adj_weight'] = pixelwise_adj_weight
    neuralef_loss_cfg['pixelwise_adj_div_factor'] = pixelwise_adj_div_factor
    neuralef_loss_cfg['pixelwise_adj_knn'] = pixelwise_adj_knn
    neuralef_loss_cfg['cache_size'] = cache_size
    dataset_cfg = cfg["dataset"][dataset]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)
    
    backbone_cfg["image_size"] = (crop_size, crop_size)
    backbone_cfg["name"] = backbone
    backbone_cfg["dropout"] = dropout

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs is not None:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        backbone_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=backbone_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=dict(
            backbone=backbone_cfg,
            psi=psi_cfg,
            kmeans=kmeans_cfg,
            neuralef=neuralef_loss_cfg,
        ),
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth"

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    net_kwargs['is_baseline'] = is_baseline
    model = create_segmenter(net_kwargs, neuralef_loss_cfg['upsample_factor'])
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    # if psi_cfg['transformer'] and optimizer_kwargs['opt'] == 'adamw':
    #     optimizer_kwargs["warmup_lr"] = 1e-6
    #     optimizer_kwargs["warmup_epochs"] = 2
    # else:
    optimizer_kwargs["warmup_lr"] = 1e-5
    optimizer_kwargs["warmup_epochs"] = 0
    optimizer_kwargs["cooldown_epochs"] = 0
    optimizer_kwargs["decay_rate"] = 1
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    if optimizer_kwargs['opt'] == 'our_lars':
        from utils.torch import LARS, exclude_bias_and_norm
        optimizer = LARS(model.parameters(),
                        lr=optimizer_kwargs['lr'], weight_decay=optimizer_kwargs['weight_decay'],
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
    else:
        optimizer = create_optimizer(opt_args, model)
    if num_epochs > 0:
        lr_scheduler = create_scheduler(opt_args, optimizer)
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_ckpt = checkpoint["model"]
        for k in list(model_ckpt.keys()):
            if 'cluster' in k or 'online_head' in k:
                if model_ckpt[k].shape[0] != kmeans_cfg['n_cls']:
                    del model_ckpt[k]
        print(model.load_state_dict(model_ckpt, strict=False))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
    else:
        sync_model(log_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    # print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Backbone parameters: {num_params(model_without_ddp.backbone)}")
    print(f"Psi parameters: {num_params(model_without_ddp.psi)}")
    print(model_without_ddp.psi)

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        if not is_baseline:
            train_logger = train_one_epoch(
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                epoch,
                amp_autocast,
                loss_scaler,
            )

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            torch.save(snapshot, checkpoint_path)
            del snapshot

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # evaluate
    tune_clusters(model, train_loader, kmeans_cfg['l2_normalize'], simulate_one_epoch=True)
    eval_logger = evaluate(
        num_epochs,
        model,
        val_loader,
        val_seg_gt,
        window_size,
        window_stride,
        amp_autocast,
        log_dir,
        is_baseline
    )
    print(f"Stats ['final']:", eval_logger, flush=True)
    print("")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
