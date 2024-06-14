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


class ClassificationEvaluator(pl.LightningModule):
    def __init__(self, weights: str = ''):
        super().__init__()
        self.weights = weights
        self.num_classes = 1000
        self.image_size = 224
        self.patch_size = 16

        # Load original weights
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(img_size=224, patch_size=16, num_classes=1000, dynamic_img_size=True).to(self.device)
        self.net.load_state_dict(state_dict, strict=True)

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


if __name__ == "__main__":
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    model = ClassificationEvaluator(weights=weights)
    # Calculate total parameters of the model
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Total parameters in net: {total_params}")

    # Calculate parameters for the patch embedding layer for image size 28
    patch_embed_params_28 = sum(p.numel() for p in model.patch_embed_2x2.parameters())
    print(f"Total parameters in patch embedding layer (image size 28): {patch_embed_params_28}")

    # Similarly, calculate for image size 224
    patch_embed_params_224 = sum(p.numel() for p in model.patch_embed_16x16.parameters())
    print(f"Total parameters in patch embedding layer (image size 224): {patch_embed_params_224}")

    from ptflops import get_model_complexity_info


    def print_flops(model, input_size):
        flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
        print(f'FLOPs for input size {input_size}: {flops}')


    # For input size 28
    print_flops(model.patch_embed_2x2, (3, 28, 28))

    # For input size 224
    print_flops(model.patch_embed_16x16, (3, 224, 224))

    from ptflops import get_model_complexity_info


    def print_flops(model, input_size):
        flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=True)
        flops_kflops = flops   # Convert to KFLOPs
        print(f'FLOPs for input size {input_size}: {flops_kflops:.2f} KFLOPs')


    # Adjust these functions to wrap your actual model operations
    def wrapped_func_28(model, x):
        # Assuming `func_28` is properly integrated in your model
        return model.patch_embed_2x2(x)


    def wrapped_func_224(model, x):
        # Assuming `func_224` is properly integrated in your model
        return model.patch_embed_16x16(x)


    # Example dummy tensor input for model
    dummy_input_28 = torch.randn(1, 3, 28, 28).to(model.device)
    dummy_input_224 = torch.randn(1, 3, 224, 224).to(model.device)

    # Calculate FLOPs
    print_flops(model.patch_embed_2x2, (3, 28, 28))
    print_flops(model.patch_embed_16x16, (3, 224, 224))

    from torchprofile import profile_macs

    inputs = torch.rand([8, 3, 224, 224])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    x = model.patch_embed_16x16.to(device)
    x.eval()
    macs = profile_macs(x, inputs)
    flops = macs / 1e9
    print(f"FLOPs:{flops}")

    inputs = torch.rand([8, 3, 28, 28])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    x = model.patch_embed_2x2.to(device)
    x.eval()
    macs = profile_macs(x, inputs)
    flops = macs / 1e9
    print(f"FLOPs:{flops}")