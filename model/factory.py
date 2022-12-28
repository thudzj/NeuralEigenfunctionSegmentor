from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

import timm
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from model.psi import MyTransformer
from model.segmenter import Segmenter
import utils.torch as ptu

from model.vit import VisionTransformer
from model.utils import checkpoint_filter_fn


def create_backbone(backbone_cfg):

    backbone_cfg = backbone_cfg.copy()
    backbone = backbone_cfg.pop("name")

    normalization = backbone_cfg.pop("normalization")
    backbone_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    backbone_cfg["d_ff"] = mlp_expansion_ratio * backbone_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        assert False

    default_cfg["input_size"] = (
        3,
        backbone_cfg["image_size"][0],
        backbone_cfg["image_size"][1],
    )
    model = VisionTransformer(**backbone_cfg)
    if "deit" in backbone or 'dino' in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn, strict=True)
    else:
        load_custom_pretrained(model, default_cfg)

    return model
    
def create_psi(backbone, psi_cfg):
    psi_cfg = psi_cfg.copy()
    psi_cfg["d_backbone"] = backbone.d_model 
    psi = MyTransformer(**psi_cfg)
    return psi


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = create_backbone(model_cfg['backbone'])
    psi = create_psi(backbone, model_cfg['psi'])
    model = Segmenter(backbone, psi, n_cls=model_cfg["n_cls"], 
        neuralef_loss_cfg=model_cfg['neuralef'], backbone_trained_by_dino=model_cfg['backbone_trained_by_dino'])
    return model

def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    print(model.load_state_dict(checkpoint, strict=False))

    return model, variant
