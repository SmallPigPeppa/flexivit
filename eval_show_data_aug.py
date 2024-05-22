import torch
import timm
import pytorch_lightning as pl
from torchinfo import summary
from thop import profile
from fvcore.nn import FlopCountAnalysis
from timm import create_model
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from flexivit_pytorch import pi_resize_patch_embed

if __name__ == "__main__":
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    # weights = 'deit3_base_patch16_224.fb_in22k_ft_in1k'
    # weights = 'pvt_v2_b3.in1k'
    net = create_model(weights, pretrained=True)
    data_config = timm.data.resolve_model_data_config(net)
    print(data_config)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    print(val_transform)
    print(train_transform)
