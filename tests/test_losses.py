import torch
from sparsechron.losses.photometric import photometric_loss
from sparsechron.losses.depth import depth_loss

def test_photometric_loss() -> None:
    # B, C, H, W format
    pred = torch.rand((1, 3, 32, 32))
    gt = torch.rand((1, 3, 32, 32))
    
    # Check L1 only
    loss_l1 = photometric_loss(pred, gt, lambda_ssim=0.0)
    assert loss_l1 > 0
    
    # Check SSIM only
    loss_ssim = photometric_loss(pred, gt, lambda_ssim=1.0)
    assert loss_ssim > 0
    
    # Check combined
    loss_combined = photometric_loss(pred, gt, lambda_ssim=0.2)
    assert loss_combined > 0

def test_depth_loss() -> None:
    # H, W, 1 format
    pred_depth = torch.ones((32, 32, 1), requires_grad=True)
    gt_depth = torch.ones((32, 32, 1)) * 2.0
    
    loss = depth_loss(pred_depth, gt_depth)
    assert loss > 0
    loss.backward()
    assert pred_depth.grad is not None
    
def test_depth_loss_empty_valid_mask() -> None:
    # H, W, 1 format
    pred_depth = torch.ones((32, 32, 1), requires_grad=True)
    # gt_depth <= 0 means invalid
    gt_depth = torch.zeros((32, 32, 1))
    
    loss = depth_loss(pred_depth, gt_depth)
    
    # Should return a zero tensor
    assert loss.item() == 0.0
    
    # Ensure it still requires grad
    assert loss.requires_grad
    
    loss.backward()
    assert pred_depth.grad is not None
    # Gradients should be zero since it's multiplied by 0.0
    assert torch.all(pred_depth.grad == 0.0)
