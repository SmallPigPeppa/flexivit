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
from flexivit_pytorch.myflex import FlexiOverlapPatchEmbed_DB as FlexiOverlapPatchEmbed


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ClassificationEvaluator(pl.LightningModule):
    def __init__(self, weights: str = ''):
        super().__init__()
        self.weights = weights
        self.num_classes = 1000
        self.image_size = 224
        self.patch_size = 16

        # Load original weights
        print(f"Loading weights {self.weights}")

        self.net = create_model(self.weights, pretrained=True)
        self.modified(new_image_size=self.image_size, new_patch_size=self.patch_size)

    def func_28(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_2x2(x, patch_size=2)
        return x

    def func_224(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_16x16(x, patch_size=16)
        return x

    def modified(self, new_image_size=224, new_patch_size=16):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        # if self.net.dynamic_img_size:
        #     self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        # self.patch_embed_3x3_s1 = self.get_new_patch_embed(new_patch_size=3, new_stride=1)
        # self.patch_embed_5x5_s2 = self.get_new_patch_embed(new_patch_size=5, new_stride=2)
        # self.patch_embed_7x7_s3 = self.get_new_patch_embed(new_patch_size=7, new_stride=3)
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

    def count_total_parameters(self):
        # Total parameters in millions
        total_params = count_parameters(self) / 1_000_000
        # Parameters of self.net in millions
        net_params = count_parameters(self.net) / 1_000_000
        # Parameters of other components in thousands
        other_params = (count_parameters(self) - count_parameters(self.net)) / 1_000

        return total_params, net_params, other_params


if __name__ == "__main__":
    weights = 'pvt_v2_b3.in1k'
    model = ClassificationEvaluator(weights=weights)

    total_params, net_params, other_params = model.count_total_parameters()
    print(f"Total Parameters: {total_params:.2f}M")
    print(f"Parameters in self.net: {net_params:.2f}M")
    print(f"Parameters in other components: {other_params:.2f}K")
