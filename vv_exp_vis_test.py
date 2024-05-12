from timm import create_model

# model_name = 'vit_base_patch16_224'
# model_name = 'deit_base_distilled_patch16_224.fb_in1k'
# model_name = 'vit_base_patch16_clip_224.openai_ft_in1k'
# model_name = 'pvt_v2_b3.in1k'
# model_name = 'mobilenetv3_small_050.lamb_in1k'
# model_name = 'resnet18.a1_in1k'
model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
import timm

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


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str
    ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resize_type: Patch embed resize method. One of ["pi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.weights = weights
        print(f"Loading weights {self.weights}")
        self.net = create_model(self.weights, pretrained=True)

    def forward(self, x):
        return self.net(x)

    def forward_patch_embed(self, x):
        a = 1
        x = self.net.patch_embed(x)
        return x

    def forward_class_token(self, x):
        x = self.net.forward_features(x)
        if self.net.attn_pool is not None:
            x = self.net.attn_pool(x)
        elif self.net.global_pool == 'avg':
            x = x[:, self.net.num_prefix_tokens:].mean(dim=1)
        elif self.net.global_pool:
            x = x[:, 0]  # class token
        return x
        # x = self.fc_norm(x)
        # x = self.head_drop(x)
        # return x if pre_logits else self.head(x)

    def forward_patch_stats(self, x):
        """Extracts the patch embeddings and computes their mean and variance."""
        patch_embeds = self.net.patch_embed(x)  # Assuming patch_embed is accessible like this
        return patch_embeds.mean(dim=[0, 1, 2]), patch_embeds.var(dim=[0, 1, 2])

    def forward_class_token_stats(self, x):
        """Extracts the class token and computes its mean and variance."""
        class_token = self.forward_class_token(x)  # Reuse your forward_class_token method
        return class_token.mean(dim=[0, 1]), class_token.var(dim=[0, 1])


if __name__ == '__main__':
    net = create_model(model_name, pretrained=True)
    print(net.default_cfg["architecture"])
    model_fn = getattr(timm.models, net.default_cfg["architecture"])
    net = model_fn(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        dynamic_img_size=True
    )

    m = ClassificationEvaluator(weights=model_name)
    x = torch.rand(size=(4, 3, 224, 224))
    patch_embeds = m.forward_patch_embed(x)
    class_tokens = m.forward_class_token(x)
    print(patch_embeds.shape)
    print(class_tokens.shape)

    patch_stats = m.forward_patch_stats(x)
    print(patch_stats)
    class_token_stats = m.forward_class_token_stats(x)
    print(class_token_stats)
