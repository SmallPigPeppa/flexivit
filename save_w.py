import torch
# from flexivit_pytorch import FlexiVisionTransformer

# net = FlexiVisionTransformer(
#     img_size=240,
#     base_patch_size=32,
#     patch_size_seq=(8, 10, 12, 15, 16, 20, 14, 30, 40, 48),
#     base_pos_embed_size=7,
#     num_classes=1000,
#     embed_dim=768,
#     depth=12,
#     num_heads=12,
#     mlp_ratio=4,
# )
#
# img = torch.randn(1, 3, 240, 240)
# preds = net(img)


# from flexivit_pytorch import (flexivit_base, flexivit_huge, flexivit_large,
#                               flexivit_small, flexivit_tiny)
#
# net = flexivit_tiny()
# net = flexivit_small()
# net = flexivit_base()
# net = flexivit_large()
# net = flexivit_huge()


from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed

from flexivit_pytorch import pi_resize_patch_embed

# Load the pretrained model's state_dict
state_dict = create_model("vit_base_patch16_224", pretrained=True).state_dict()

# Resize the patch embedding
# new_patch_size = (32, 32)
new_patch_size = (16, 16)
# state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
#     patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=new_patch_size
# )

# # Interpolate the position embedding size
image_size = 224
# grid_size = image_size // new_patch_size[0]
# state_dict["pos_embed"] = resample_abs_pos_embed(
#     posemb=state_dict["pos_embed"], new_size=[grid_size, grid_size]
# )

# Load the new weights into a model with the target image and patch sizes
net = create_model(
    "vit_base_patch16_224", img_size=image_size, patch_size=new_patch_size
)
net.load_state_dict(state_dict, strict=True)

# 指定要保存模型参数的文件路径
model_path = 'vit_base_patch16_224.pth'

# 保存模型参数
torch.save(net.state_dict(), model_path)

print(f'Model parameters saved to {model_path}')

