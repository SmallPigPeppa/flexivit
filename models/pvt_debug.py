import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import torch
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
from flexivit_pytorch.utils import resize_abs_pos_embed
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from timm.models._manipulate import checkpoint_seq

from timm.layers import PatchEmbed
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
import random
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from flexivit_pytorch.myflex import FlexiOverlapPatchEmbed


class ClassificationEvaluator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = 'pvt_v2_b3.in1k'
        self.net = create_model(weights, pretrained=True)
        self.modified()

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.net.forward_head(x)
        return x

    def forward_after_patch_embed(self, x):
        x = self.net.stages(x)
        x = self.net.forward_head(x)
        return x

    def ms_forward(self, x):
        x_3x3 = F.interpolate(x, size=56, mode='bilinear')
        x_3x3 = self.patch_embed_3x3_s1(x_3x3, patch_size=3, stride=1)

        x_5x5 = F.interpolate(x, size=112, mode='bilinear')
        x_5x5 = self.patch_embed_5x5_s2(x_5x5, patch_size=5, stride=2)

        x_7x7 = F.interpolate(x, size=168, mode='bilinear')
        x_7x7 = self.patch_embed_7x7_s3(x_7x7, patch_size=7, stride=3)

        x_7x7_s4 = F.interpolate(x, size=224, mode='bilinear')
        x_7x7_s4 = self.patch_embed_7x7_s4(x_7x7_s4, patch_size=7, stride=4)

        return self.forward_after_patch_embed(x_3x3), \
            self.forward_after_patch_embed(x_5x5), \
            self.forward_after_patch_embed(x_7x7), \
            self.forward_after_patch_embed(x_7x7_s4)

    def rand_ms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # 随机选择imagesize
        img_size_3x3 = random.choice([28, 42, 56, 70, 84])
        x_3x3 = F.interpolate(x, size=(img_size_3x3, img_size_3x3), mode='bilinear')
        x_3x3 = self.patch_embed_3x3_s1(x_3x3, patch_size=3, stride=1)

        img_size_5x5 = random.choice([84, 98, 112, 126, 140])
        x_5x5 = F.interpolate(x, size=(img_size_5x5, img_size_5x5), mode='bilinear')
        x_5x5 = self.patch_embed_5x5_s2(x_5x5, patch_size=5, stride=2)

        img_size_7x7 = random.choice([140, 154, 168, 182, 196])
        x_7x7 = F.interpolate(x, size=(img_size_7x7, img_size_7x7), mode='bilinear')
        x_7x7 = self.patch_embed_7x7_s3(x_7x7, patch_size=7, stride=3)

        img_size_7x7_s4 = random.choice([196, 210, 224, 238, 252])
        x_7x7_s4 = F.interpolate(x, size=(img_size_7x7_s4, img_size_7x7_s4), mode='bilinear')
        x_7x7_s4 = self.patch_embed_7x7_s4(x_7x7_s4, patch_size=7, stride=4)

        return self.forward_after_patch_embed(x_3x3), \
            self.forward_after_patch_embed(x_5x5), \
            self.forward_after_patch_embed(x_7x7), \
            self.forward_after_patch_embed(x_7x7_s4)

    def ms_forward_debug(self, x):
        x = self.patch_embed_3x3_s1(x, patch_size=3, stride=1)
        x = self.net.stages(x)
        x = self.net.forward_head(x)
        return x

    def modified(self):
        self.in_chans = 3
        self.embed_dim = 64
        self.patch_embed_3x3_s1 = self.get_new_patch_embed(new_patch_size=3, new_stride=1)
        self.patch_embed_5x5_s2 = self.get_new_patch_embed(new_patch_size=5, new_stride=2)
        self.patch_embed_7x7_s3 = self.get_new_patch_embed(new_patch_size=7, new_stride=3)
        self.patch_embed_7x7_s4 = self.get_new_patch_embed(new_patch_size=7, new_stride=4)

    def get_new_patch_embed(self, new_patch_size, new_stride):
        new_patch_embed = FlexiOverlapPatchEmbed(
            patch_size=new_patch_size,
            stride=new_stride,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            new_weight = pi_resize_patch_embed(
                patch_embed=origin_weight, new_patch_size=(new_patch_size, new_patch_size)
            )
            new_patch_embed.proj.weight = nn.Parameter(new_weight, requires_grad=True)
        if self.net.patch_embed.proj.bias is not None:
            new_patch_embed.proj.bias = nn.Parameter(self.net.patch_embed.proj.bias.clone().detach(),
                                                     requires_grad=True)

        return new_patch_embed


m = ClassificationEvaluator()
# x = torch.rand(size=(16, 3, 56, 56))
# x1 = m.patch_embed_3x3_s1(x, patch_size=3, stride=1)
# print(x1.shape)
#
# x = torch.rand(size=(16, 3, 224, 224))
# x2 = m.net.patch_embed(x)
# print(x2.shape)


x = torch.rand(size=(1, 3, 224, 224))
x3 = m.ms_forward(x)
print(len(x3))
print(x3[0].shape)

# x = torch.rand(size=(1, 3, 56, 56))
# x4 = m.ms_forward(x)
# print(x4.shape)
#
# x = torch.rand(size=(1, 3, 112, 112))
# x5 = m.ms_forward(x)
# print(x5.shape)


