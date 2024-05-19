import torch
import torch.nn.functional as F

x = torch.rand(size=(8, 3, 224, 224))
print(x.shape)
new_size = (128, 128)
interpolation = 'bicubic'
interpolation = 'nearest'
interpolation = 'bilinear'
interpolation = 'area'
patch_embed = F.interpolate(
    x, new_size, mode=interpolation
)

print(patch_embed.shape)
