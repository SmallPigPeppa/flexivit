import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor
from flexivit_pytorch.utils import to_2tuple
from typing import Optional, Sequence, Tuple, Union, List


class FlexiOverlapPatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 7,
            stride: Union[int, Tuple[int, int]] = 4,
            in_chans: int = 3,
            embed_dim: int = 768,
            bias: bool = True,
            interpolation: str = "bicubic",
            antialias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            bias=bias,
        )
        self.interpolation = interpolation
        self.antialias = antialias
        self.norm = nn.LayerNorm(embed_dim)
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        self.patch_size_seq = [self.patch_size]
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
            stride: int = None,
    ) -> Tensor:

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight

        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=stride,
                     padding=(patch_size[0] // 2, patch_size[1] // 2))

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class FlexiMViTPatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            patch_size=7,
            stride=4,
            in_chans=3,
            embed_dim=768,
            bias=True,
            interpolation="bicubic",
            antialias=True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            bias=bias,
        )
        self.interpolation = interpolation
        self.antialias = antialias
        self.norm = nn.LayerNorm(embed_dim)
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        self.patch_size_seq = [self.patch_size]
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

    def forward(self, x, patch_size, stride, ) -> Tuple[torch.Tensor, List[int]]:
        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight

        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=stride,
                     padding=(patch_size[0] // 2, patch_size[1] // 2))
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape[-2:]
