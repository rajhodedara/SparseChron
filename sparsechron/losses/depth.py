import torch

def depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    """Computes the L1 loss for valid depth pixels.
    
    Args:
        pred_depth: The predicted depth tensor (H, W, 1) or (1, H, W).
        gt_depth: The ground truth depth tensor.
        
    Returns:
        The mean L1 loss on valid pixels, or 0.0 with gradients if no valid pixels.
    """
    assert isinstance(pred_depth, torch.Tensor), "pred_depth must be a torch.Tensor"
    assert isinstance(gt_depth, torch.Tensor), "gt_depth must be a torch.Tensor"
    
    pred_depth = pred_depth.squeeze(-1)
    gt_depth = gt_depth.squeeze(-1)
    
    valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    if not valid_mask.any():
        return (pred_depth * 0.0).sum()
        
    return torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask]).mean()
