import torch
from flexivit_pytorch import FlexiVisionTransformer


from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed

from flexivit_pytorch import pi_resize_patch_embed

# Load the pretrained model's state_dict
state_dict = create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True).state_dict()

# Resize the patch embedding
new_patch_size = (32, 32)
state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
    patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=new_patch_size
)

# Interpolate the position embedding size
image_size = 224
grid_size = image_size // new_patch_size[0]
state_dict["pos_embed"] = resample_abs_pos_embed(
    posemb=state_dict["pos_embed"], new_size=[grid_size, grid_size]
)

# Load the new weights into a model with the target image and patch sizes
net = create_model(
    "vit_base_patch16_224.augreg2_in21k_ft_in1k", img_size=image_size, patch_size=new_patch_size
)
net.load_state_dict(state_dict, strict=True)

print(net)


