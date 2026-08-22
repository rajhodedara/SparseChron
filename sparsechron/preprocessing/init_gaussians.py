"""Gaussian initialization from point clouds."""
import math
import torch
from typing import Dict


def initialize_gaussians(
    points: torch.Tensor, colors: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Initializes 3D Gaussian parameters from a point cloud.

    Args:
        points: (N, 3) tensor of point cloud positions.
        colors: (N, 3) tensor of RGB colors in [0, 1].

    Returns:
        Dictionary containing initialized Gaussian parameters:
            - positions: (N, 3) tensor of positions.
            - scales: (N, 3) tensor of scales in log-space.
            - rotations: (N, 4) tensor of identity quaternions.
            - opacities: (N, 1) tensor of inverse-sigmoid opacities.
            - sh_coeffs: (N, 1, 3) tensor of spherical harmonics DC coefficients.
            
    Raises:
        ValueError: If input shapes are incorrect.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same number of elements")
    if points.device != colors.device:
        raise ValueError("points and colors must be on the same device")
    
    N = points.shape[0]
    if N == 0:
        raise ValueError("points and colors cannot be empty")

    # Scales: distance to 3 nearest neighbors (excluding self)
    if N > 1:
        k = min(N, 4)
        mean_dists_list = []
        chunk_size = 5000
        for i in range(0, N, chunk_size):
            chunk_points = points[i:i + chunk_size]
            dists = torch.cdist(chunk_points, points)
            nn_dists, _ = dists.topk(k, largest=False)
            mean_dists_list.append(nn_dists[:, 1:].mean(dim=1, keepdim=True))
        mean_dists = torch.cat(mean_dists_list, dim=0)
    else:
        # Fallback if only 1 point
        mean_dists = torch.ones((1, 1), dtype=points.dtype, device=points.device)

    # Broadcast to (N, 3) and apply log
    scales = mean_dists.clamp_min(1e-7).expand(N, 3).log()

    # Rotations: [1, 0, 0, 0]
    rotations = torch.zeros((N, 4), dtype=points.dtype, device=points.device)
    rotations[:, 0] = 1.0

    # Opacities: logit(0.1)
    # logit(x) = log(x / (1 - x))
    inverse_sigmoid_01 = math.log(0.1 / (1.0 - 0.1))
    opacities = torch.full(
        (N, 1), inverse_sigmoid_01, dtype=points.dtype, device=points.device
    )

    # SH Coeffs: initialized using logit to work with sigmoid activation
    colors_clamped = colors.clamp(min=1e-4, max=1.0-1e-4)
    # 3rd degree SH has 16 coefficients. First is DC, rest are 0.
    sh_coeffs = torch.zeros((N, 16, 3), dtype=points.dtype, device=points.device)
    sh_coeffs[:, 0, :] = torch.log(colors_clamped / (1.0 - colors_clamped))

    return {
        "positions": points.clone(),
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }
