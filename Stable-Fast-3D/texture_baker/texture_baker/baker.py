from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# CPU-only extension (common on Windows without CUDA toolkit at build time) while the rest of the pipeline uses CUDA.
_texture_baker_force_cpu: bool | None = None


class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()

    def rasterize(
        self,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Rasterize the UV coordinates to a barycentric coordinates
        & Triangle idxs texture map

        Args:
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 4, float: Rasterized map
        """
        global _texture_baker_force_cpu

        dev = uv.device
        fi = face_indices.to(torch.int32)

        def _call(u: Tensor, f: Tensor) -> Tensor:
            return torch.ops.texture_baker_cpp.rasterize(u, f, bake_resolution)

        if dev.type == "cpu":
            return _call(uv, fi)

        if _texture_baker_force_cpu is True:
            return _call(uv.cpu(), fi.cpu()).to(dev)
        if _texture_baker_force_cpu is False:
            return _call(uv, fi)

        try:
            out = _call(uv, fi)
            _texture_baker_force_cpu = False
            return out
        except NotImplementedError:
            _texture_baker_force_cpu = True
            return _call(uv.cpu(), fi.cpu()).to(dev)

    def get_mask(self, rast: Tensor) -> Tensor:
        """
        Get the occupancy mask from the rasterized map

        Args:
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map

        Returns:
            Tensor, bake_resolution bake_resolution, bool: Mask
        """
        return rast[..., -1] >= 0

    def interpolate(
        self,
        attr: Tensor,
        rast: Tensor,
        face_indices: Tensor,
    ) -> Tensor:
        """
        Interpolate the attributes using the rasterized map

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Interpolated attributes
        """
        global _texture_baker_force_cpu

        dev = attr.device
        fi = face_indices.to(torch.int32)

        def _call(a: Tensor, r: Tensor, f: Tensor) -> Tensor:
            return torch.ops.texture_baker_cpp.interpolate(a, f, r)

        if dev.type == "cpu":
            return _call(attr, rast, fi)

        if _texture_baker_force_cpu is True:
            return _call(attr.cpu(), rast.cpu(), fi.cpu()).to(dev)
        if _texture_baker_force_cpu is False:
            return _call(attr, rast, fi)

        try:
            out = _call(attr, rast, fi)
            _texture_baker_force_cpu = False
            return out
        except NotImplementedError:
            _texture_baker_force_cpu = True
            return _call(attr.cpu(), rast.cpu(), fi.cpu()).to(dev)

    def forward(
        self,
        attr: Tensor,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Bake the texture

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Baked texture
        """
        rast = self.rasterize(uv, face_indices, bake_resolution)
        return self.interpolate(attr, rast, face_indices)
