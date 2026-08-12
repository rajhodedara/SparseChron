"""Depth map alignment."""
import torch
from typing import List


def align_depths(
    relative_depths: List[torch.Tensor],
    metric_depths: List[torch.Tensor],
    valid_masks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Aligns relative monocular depths to metric sparse depths using least squares.

    Args:
        relative_depths: List of (H, W) tensors containing relative depths.
        metric_depths: List of (H, W) tensors containing metric depths.
        valid_masks: List of (H, W) boolean tensors indicating valid metric depth pixels.

    Returns:
        List of (H, W) tensors containing the aligned depths.
        
    Raises:
        ValueError: If input lists do not have the same length.
    """
    if len(relative_depths) != len(metric_depths) or len(relative_depths) != len(
        valid_masks
    ):
        raise ValueError("All input lists must have the same length.")

    aligned_depths = []

    for rel_d, met_d, mask in zip(relative_depths, metric_depths, valid_masks):
        if not (rel_d.shape == met_d.shape == mask.shape):
            raise ValueError("rel_d, met_d, and mask must have the same shape")
        
        mask = mask.to(torch.bool)
        
        # Flatten the valid pixels
        valid_rel = rel_d[mask]
        valid_met = met_d[mask]

        if valid_rel.shape[0] < 2:
            # Not enough points to fit scale and shift, return unchanged
            aligned_depths.append(rel_d.clone())
            continue

        # We want to solve for s and t in: s * valid_rel + t = valid_met
        # A * [s, t]^T = valid_met
        # where A = [valid_rel, 1]
        A = torch.stack((valid_rel, torch.ones_like(valid_rel)), dim=-1)

        try:
            # Least squares solve
            # torch.linalg.lstsq solves AX = B
            solution = torch.linalg.lstsq(A, valid_met).solution
            s, t = solution[0], solution[1]

            # Avoid NaNs or completely invalid solutions
            if not (torch.isfinite(s).item() and torch.isfinite(t).item()):
                aligned_depths.append(rel_d.clone())
            else:
                aligned = s * rel_d + t
                aligned_depths.append(aligned)
        except RuntimeError:
            # Fallback if lstsq fails
            aligned_depths.append(rel_d.clone())

    return aligned_depths
