import sys, os
from pathlib import Path
import yaml
import json
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
from engine import train_one_epoch, evaluate, reco_protocal_eval

import warnings
warnings.filterwarnings('ignore')

@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--port", default=None, type=int)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_small_patch16_384", type=str)
@click.option("--optimizer", default="adam", type=str)
@click.option("--scheduler", default="cosine", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)

@click.option("--eval-only/--no-eval-only", default=False, is_flag=True)
@click.option("--reco/--no-reco", default=False, is_flag=True)

@click.option("--beta", default=None, type=float)
@click.option("--recon_target", default=None, type=str)

@click.option("--encoder_input_type", default=None, type=str)
@click.option("--encoder_d_model", default=None, type=int)
@click.option("--encoder_residual/--no-encoder_residual", default=False, is_flag=True)

@click.option("--embedding_dim", default=None, type=int)
@click.option("--n_embeddings", default=None, type=int)

@click.option("--decoder_n_layers", default=None, type=int)
@click.option("--decoder_d_model", default=None, type=int)
@click.option("--decoder_n_heads", default=None, type=int)

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
    eval_only,
    reco,
    beta,
    recon_target,
    encoder_input_type,
    encoder_d_model,
    encoder_residual,
    embedding_dim,
    n_embeddings,
    decoder_n_layers,
    decoder_d_model,
    decoder_n_heads
):
    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process(port)

    if eval_only:
        resume = True

    # set up configuration
    cfg = config.load_config()
    backbone_cfg = cfg["backbone"][backbone]
    encoder_cfg = cfg["encoder"]
    decoder_cfg = cfg["decoder"]
    dataset_cfg = cfg["dataset"][dataset]
    loss_cfg = cfg["loss"]

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

    if beta is not None:
        loss_cfg['beta'] = beta
    if recon_target is not None:
        loss_cfg['recon_target'] = recon_target

    if encoder_input_type is not None:
        encoder_cfg['input_type'] = encoder_input_type
    if encoder_d_model is not None:
        encoder_cfg['d_model'] = encoder_d_model
    encoder_cfg['residual'] = encoder_residual
    if embedding_dim is not None:
        encoder_cfg['embedding_dim'] = embedding_dim
    if n_embeddings is not None:
        encoder_cfg['n_embeddings'] = n_embeddings
    
    if decoder_n_layers is not None:
        decoder_cfg['n_layers'] = decoder_n_layers
    if decoder_d_model is not None:
        decoder_cfg['d_model'] = decoder_d_model
    if decoder_n_heads is not None:
        decoder_cfg['n_heads'] = decoder_n_heads

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
            encoder=encoder_cfg,
            decoder=decoder_cfg,
            backbone=backbone_cfg,
            backbone_trained_by_dino='dino' in backbone,
        ),
        loss_kwargs=loss_cfg,
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
    loss_kwargs = variant["loss_kwargs"]
    net_kwargs["n_cls"] = n_cls
    model = create_segmenter(net_kwargs, loss_kwargs)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    optimizer_kwargs["warmup_lr"] = 1e-5
    optimizer_kwargs["warmup_epochs"] = 0
    optimizer_kwargs["cooldown_epochs"] = 0
    optimizer_kwargs["decay_rate"] = 1
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, [p for p in model.parameters() if p.requires_grad])
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
        print(model.load_state_dict(model_ckpt, strict=False))
        if not eval_only:
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

    if dataset != 'imagenet':
        val_seg_gt = val_loader.dataset.get_gt_seg_maps()
    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Backbone parameters: {num_params(model_without_ddp.backbone)}")
    print(f"Backbone parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Backbone parameters: {num_params(model_without_ddp.decoder)}")
    print(model_without_ddp.encoder)
    print(model_without_ddp.decoder)

    for epoch in range(start_epoch, num_epochs):
        if eval_only:
            break

        # train for one epoch
        train_logger = train_one_epoch(
            num_epochs,
            dataset,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            log_dir,
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

    if reco:
        if dataset == 'imagenet':
            reco_protocal_eval(model, 'cityscapes', 'val', backbone_cfg["normalization"], log_dir, n_cls_train=encoder_cfg['n_embeddings'])
            reco_protocal_eval(model, 'pascal_context', 'val', backbone_cfg["normalization"], log_dir, n_cls_train=encoder_cfg['n_embeddings'])
        else:
            reco_protocal_eval(model, dataset, 'val', backbone_cfg["normalization"], log_dir, n_cls_train=encoder_cfg['n_embeddings'])
    else:
        if dataset == 'imagenet':
            raise NotImplementedError
        else:
            eval_logger = evaluate(
                dataset,
                num_epochs,
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
                log_dir,
                n_cls,
                n_cls_train=encoder_cfg['n_embeddings']
            )
            print(f"Stats ['final']:", eval_logger, flush=True)
            print("")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
