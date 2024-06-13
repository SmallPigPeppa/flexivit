import torch
import timm
from torchinfo import summary
from timm import create_model
import torch.nn as nn
from models.flex_patch_embed import FlexiPatchEmbed
from flexivit_pytorch import pi_resize_patch_embed
import pytorch_lightning as pl
from torchprofile import profile_macs

class ClassificationEvaluator(pl.LightningModule):
    def __init__(self, weights: str = ''):
        super().__init__()
        self.weights = weights
        self.image_size = 224  # Default image size
        self.patch_size = 16  # Default patch size

        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        self.net = orig_net.to(self.device)
        self.patch_embed = nn.Identity()  # Placeholder for dynamic patch embedding

    def get_new_patch_embed(self, new_image_size, new_patch_size):
        # 创建新的patch embedding layer
        new_patch_embed = FlexiPatchEmbed(
            img_size=new_image_size,
            patch_size=new_patch_size,
            in_chans=3,
            embed_dim=self.net.num_features,
            bias=True,
        )
        return new_patch_embed

    def calculate_flops(self, image_size):
        patch_size = image_size // 14
        patch_embed = self.get_new_patch_embed(new_image_size=image_size, new_patch_size=patch_size).to(self.device)

        # Generate a dummy input based on the image size
        dummy_input = torch.randn(8, 3, image_size, image_size).to(self.device)
        patch_embed.eval()

        # Calculate MACs (Multiply-Accumulate operations)
        macs = profile_macs(patch_embed, dummy_input)
        flops = macs   # Convert MACs to FLOPs (2 operations per MAC)
        return flops / 1e9  # Convert to GFLOPs

if __name__ == "__main__":
    weights = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    model = ClassificationEvaluator(weights=weights).to('cpu')

    image_sizes = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224,336, 448]
    flops_results = []


    for size in image_sizes:
        flops = model.calculate_flops(size)
        flops_results.append(f"{flops:.4f}")  # Format flops to four decimal places

    print("FLOPs results:", flops_results)


