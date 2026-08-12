import torch
import torch.nn.functional as F


def texture_regularization_loss(
    d_pos: torch.Tensor,
    projected_points: torch.Tensor,
    gt_image: torch.Tensor,
    beta: float = 10.0,
) -> torch.Tensor:
    """Computes a texture-aware regularization loss for deformations.

    Penalizes deformations more strongly in texture-less regions.

    Args:
        d_pos (torch.Tensor): Deformations of dynamic Gaussians (N, 3).
        projected_points (torch.Tensor): 2D screen coordinates of Gaussians (N, 2).
        gt_image (torch.Tensor): Ground truth image (C, H, W).
        beta (float): Parameter controlling the steepness of the penalty weight.

    Returns:
        torch.Tensor: The texture-aware regularization loss.
    """
    if d_pos.numel() == 0:
        return torch.tensor(0.0, device=d_pos.device, requires_grad=True)

    C, H, W = gt_image.shape

    # Compute image gradients using finite differences
    # Compute differences along x and y (padding to keep size same)
    diff_y = gt_image[:, 1:, :] - gt_image[:, :-1, :]
    diff_x = gt_image[:, :, 1:] - gt_image[:, :, :-1]

    # Pad to original size
    grad_y = F.pad(diff_y, (0, 0, 0, 1))
    grad_x = F.pad(diff_x, (0, 1, 0, 0))

    # Gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    # Mean across channels
    grad_mag = grad_mag.mean(dim=0, keepdim=True)  # (1, H, W)
    grad_mag = grad_mag.unsqueeze(0)  # (1, 1, H, W) for grid_sample

    # Normalize projected points to [-1, 1]
    # projected_points is (N, 2) where col 0 is X, col 1 is Y
    x_norm = 2.0 * (projected_points[:, 0] / max(W - 1, 1)) - 1.0
    y_norm = 2.0 * (projected_points[:, 1] / max(H - 1, 1)) - 1.0
    
    # Grid sample expects shape (N, 1, 1, 2) or (1, 1, N, 2)
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (N, 2)
    grid = grid.view(1, 1, -1, 2)  # (1, 1, N, 2)

    # Sample gradient magnitude
    sampled_grad = F.grid_sample(
        grad_mag, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (1, 1, 1, N)
    sampled_grad = sampled_grad.view(-1)  # (N,)

    # Compute weights
    weights = torch.exp(-beta * sampled_grad)

    # Compute loss
    d_pos_sq = (d_pos**2).sum(dim=-1)  # (N,)
    loss = (weights * d_pos_sq).mean()

    return loss
