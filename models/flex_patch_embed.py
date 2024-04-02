from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor

from flexivit_pytorch.utils import to_2tuple
from timm.layers.format import Format, nchw_to


class FlexiPatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 240,
            patch_size: Union[int, Tuple[int, int]] = 32,
            grid_size: Union[int, Tuple[int, int]] = 7,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[nn.Module] = None,
            flatten: bool = True,
            bias: bool = True,
            patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
            patch_size_probs: Optional[Sequence[float]] = None,
            interpolation: str = "bicubic",
            antialias: bool = True,
            output_fmt: Optional[str] = None,
            strict_img_size=False
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24

        Args:
            img_size: Input image size
            patch_size: Base patch size. i.e the size of the parameter buffer
            grid_size: Size of pos_embed buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            flatten: Whether to flatten the spatial dimensions of the output
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            patch_size_probs: Optional list of probabilities to sample corresponding
                patch_size_seq elements. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = to_2tuple(grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        if self.patch_size_seq:
            if not patch_size_probs:
                n = len(self.patch_size_seq)
                self.patch_size_probs = [1.0 / n] * n
            else:
                self.patch_size_probs = [
                    p / sum(patch_size_probs) for p in patch_size_probs
                ]
        else:
            self.patch_size_probs = []

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            ps = to_2tuple(ps)
            pinvs[ps] = self._calculate_pinv(self.patch_size, ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
            self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                self.patch_size, new_patch_size
            )
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
            self,
            x: Tensor,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            return_patch_size: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:

        if not patch_size and not self.training:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size
        elif not patch_size:
            # During training choose uniformly at random if not specified
            assert (
                self.patch_size_seq
            ), "No patch size specified during forward and no patch_size_seq given to FlexiPatchEmbed"
            patch_size = np.random.choice(self.patch_size_seq, p=self.patch_size_probs)

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)

        x = self.norm(x)

        if return_patch_size:
            return x, patch_size

        return x


if __name__ == '__main__':
    # def modified(self, new_image_size=224, new_patch_size=16):
    embed_args = {}
    sin_chans = 3
    embed_dim = 784
    pre_norm = False
    dynamic_img_pad = False
    dynamic_img_size = True
    if dynamic_img_size:
        # flatten deferred until after pos embed
        embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
    # self.patch_embed_4x4 = self.get_new_patch_embed(new_image_size=56, new_patch_size=4)
    # self.patch_embed_8x8 = self.get_new_patch_embed(new_image_size=112, new_patch_size=8)
    # self.patch_embed_16x16 = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
    # self.patch_embed_16x16_origin = self.get_new_patch_embed(new_image_size=224, new_patch_size=16)
    new_image_size = 56
    new_patch_size = 4
    in_chans = 3

    # self.net.patch_embed = nn.Identity()

    # def get_new_patch_embed(self, new_image_size, new_patch_size):
    new_patch_embed = FlexiPatchEmbed(
        img_size=new_image_size,
        patch_size=new_patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        # dynamic_img_pad=self.dynamic_img_pad,
        **embed_args,
    )
    x = torch.rand(size=[4, 3, 224, 224])
    pe = new_patch_embed(x, patch_size=16)
    print(pe.shape)
    x = torch.rand(size=[4, 3, 112, 112])
    pe = new_patch_embed(x, patch_size=8)
    print(pe.shape)

    x = torch.rand(size=[4, 3, 112, 224])
    pe = new_patch_embed(x, patch_size=[8, 16])
    print(pe.shape)
