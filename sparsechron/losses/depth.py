import torch

def depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    """Computes the scale-and-shift invariant depth loss (Pearson Correlation).
    
    Args:
        pred_depth: The predicted depth tensor (H, W, 1) or (1, H, W).
        gt_depth: The ground truth depth tensor.
        
    Returns:
        The Pearson correlation-based loss on valid pixels.
    """
    assert isinstance(pred_depth, torch.Tensor), "pred_depth must be a torch.Tensor"
    assert isinstance(gt_depth, torch.Tensor), "gt_depth must be a torch.Tensor"
    
    pred_depth = pred_depth.squeeze(-1)
    gt_depth = gt_depth.squeeze(-1)
    
    valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth)
    if not valid_mask.any():
        return (pred_depth * 0.0).sum()
        
    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]
    
    pred_centered = pred - pred.mean()
    gt_centered = gt - gt.mean()
    
    cov = (pred_centered * gt_centered).mean()
    pred_var = (pred_centered ** 2).mean().clamp_min(1e-6)
    gt_var = (gt_centered ** 2).mean().clamp_min(1e-6)
    
    corr = cov / torch.sqrt(pred_var * gt_var)
    return 1.0 - corr
