from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

import timm
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from model.custom_models import Encoder, Decoder
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
    
def create_encoder(backbone, encoder_cfg):
    encoder_cfg = encoder_cfg.copy()
    encoder_cfg["d_backbone"] = backbone.d_model 
    encoder_cfg["patch_size"] = backbone.patch_embed.patch_size
    encoder = Encoder(**encoder_cfg)
    return encoder

def create_decoder(backbone, encoder_cfg, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    decoder_cfg["image_size"] = backbone.patch_embed.image_size
    decoder_cfg["d_backbone"] = backbone.d_model
    decoder_cfg["embedding_dim"] = encoder_cfg['embedding_dim']
    decoder_cfg["patch_size"] = backbone.patch_embed.patch_size
    decoder = Decoder(**decoder_cfg)
    return decoder

def create_segmenter(model_cfg, loss_cfg):
    model_cfg = model_cfg.copy()
    backbone = create_backbone(model_cfg['backbone'])
    encoder = create_encoder(backbone, model_cfg['encoder'])
    decoder = create_decoder(backbone, model_cfg['encoder'], model_cfg['decoder'])
    model = Segmenter(backbone, encoder, decoder, n_cls=model_cfg["n_cls"], 
        loss_cfg=loss_cfg, backbone_trained_by_dino=model_cfg['backbone_trained_by_dino'])
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
