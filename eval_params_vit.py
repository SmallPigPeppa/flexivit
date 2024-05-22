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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ClassificationEvaluator(pl.LightningModule):
    def __init__(self, weights: str = ''):
        super().__init__()
        self.weights = weights
        # Load original weights
        print(f"Loading weights {self.weights}")
        self.net = create_model(self.weights, pretrained=True)
        self.modified()

    def func_28(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_2x2(x, patch_size=2)
        return x

    def func_224(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed_16x16(x, patch_size=16)
        return x

    def modified(self):
        self.embed_args = {}
        self.in_chans = 3
        self.embed_dim = self.net.num_features
        self.pre_norm = False
        self.dynamic_img_pad = False
        if self.net.dynamic_img_size:
            self.embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed_4x4 = self.get_new_patch_embed(new_image_size=56, new_patch_size=4)
        self.patch_embed_8x8 = self.get_new_patch_embed(new_image_size=112, new_patch_size=8)
        self.patch_embed_12x12 = self.get_new_patch_embed(new_image_size=168, new_patch_size=12)
        self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        # self.patch_embed_16x16_origin = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
        self.net.patch_embed = nn.Identity()

    def get_new_patch_embed(self, new_image_size, new_patch_size):
        new_patch_embed = FlexiPatchEmbed(
            img_size=new_image_size,
            patch_size=new_patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,
            **self.embed_args,
        )
        if hasattr(self.net.patch_embed.proj, 'weight'):
            origin_weight = self.net.patch_embed.proj.weight.clone().detach()
            new_weight = pi_resize_patch_embed(patch_embed=origin_weight,
                                               new_patch_size=(new_patch_size, new_patch_size))
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
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    weights = 'deit3_base_patch16_224.fb_in22k_ft_in1k'
    model = ClassificationEvaluator(weights=weights)

    total_params, net_params, other_params = model.count_total_parameters()
    print(f"Total Parameters: {total_params:.2f}M")
    print(f"Parameters in self.net: {net_params:.2f}M")
    print(f"Parameters in other components: {other_params:.2f}K")
