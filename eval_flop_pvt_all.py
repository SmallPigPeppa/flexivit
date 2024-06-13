import torch
import timm
import pytorch_lightning as pl
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from timm import create_model
import torch.nn as nn
from flexivit_pytorch import pi_resize_patch_embed
from flexivit_pytorch.myflex import FlexiOverlapPatchEmbed

from torchprofile import profile_macs


class ClassificationEvaluator(pl.LightningModule):
    def __init__(self, weights: str = '', base_image_size=224, base_patch_size=16):
        super().__init__()
        self.weights = weights
        self.base_image_size = base_image_size
        self.base_patch_size = base_patch_size
        self.num_classes = 1000

        # Load original weights and model
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        self.net = create_model(self.weights, pretrained=True, img_size=self.base_image_size,
                                num_classes=self.num_classes, dynamic_img_size=True).to(self.device)
        self.net.load_state_dict(orig_net.state_dict(), strict=True)

    def configure_patch_embed(self, image_size):
        scale = image_size / self.base_image_size
        patch_size = int(self.base_patch_size * scale)
        stride = patch_size  # Assume stride equals patch size
        return FlexiOverlapPatchEmbed(patch_size=patch_size, stride=stride, in_chans=3, embed_dim=self.net.num_features)


if __name__ == "__main__":
    weights = 'vit_base_patch16_224'
    model = ClassificationEvaluator(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    resolutions = [28, 42, 56, 70, 84, 98, 112, 126, 140, 168, 224, 336, 448]
    flops_results = []

    for res in resolutions:
        patch_embed = model.configure_patch_embed(res)
        inputs = torch.rand([8, 3, res, res]).to(device)
        patch_embed.to(device).eval()

        macs = profile_macs(patch_embed.proj, inputs)
        flops = macs / 1e9  # Convert to Giga FLOPs
        flops_results.append(f"{flops:.4f}")

    print("FLOPs results:", flops_results)

