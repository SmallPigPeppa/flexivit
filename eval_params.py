import torch
import pytorch_lightning as pl
import timm.models
from timm import create_model
from flexivit_pytorch import (interpolate_resize_patch_embed, pi_resize_patch_embed)
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from fvcore.nn import FlopCountAnalysis
from timm import create_model
import torch
import torch.nn as nn
from functools import partial
from torchinfo import summary
from thop import profile


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str = '',
    ):
        super().__init__()
        self.weights = weights
        self.num_classes = 1000
        self.image_size = 224
        self.patch_size = 16

        # Load original weights
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()
        self.origin_state_dict = state_dict
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=224,
            patch_size=16,
            num_classes=1000,
            dynamic_img_size=True
        ).to(self.device)
        self.net.load_state_dict(state_dict, strict=True)

        # modified
        self.modified(new_image_size=self.image_size, new_patch_size=self.patch_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_2x2(x, patch_size=2)
        return x

    def forward_28(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_2x2(x, patch_size=2)
        return x

    def forward_224(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_16x16(x, patch_size=16)
        return x

    def modified(self, new_image_size=224, new_patch_size=16):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed_2x2 = self.get_new_patch_embed(new_image_size=28, new_patch_size=2)
        self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        self.net.patch_embed = nn.Identity()

    def get_new_patch_embed(self, new_image_size, new_patch_size):
        new_patch_embed = FlexiPatchEmbed(
            img_size=new_image_size,
            patch_size=new_patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **self.embed_args,
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


def analyze_model(model, input_size=(256, 3, 28, 28)):
    # 为模型生成一个随机输入
    inputs = torch.randn(input_size)

    # 使用thop计算FLOPs和参数
    flops, params = profile(model, inputs=(inputs,), verbose=False)

    # 打印模型的详细信息
    print(summary(model, input_size=input_size))

    # 返回计算结果
    return flops, params






if __name__ == "__main__":
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    model = ClassificationEvaluator(weights=weights)
    # 分析TIMM模型
    flops_timm, params_timm = analyze_model(model)
    print(f"TIMM model - FLOPs: {flops_timm}, Params: {params_timm}")

