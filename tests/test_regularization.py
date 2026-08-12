import torch
import torch.nn.functional as F
from sparsechron.losses.regularization import texture_regularization_loss

def test_texture_regularization_loss_empty_d_pos():
    """Test with empty d_pos."""
    d_pos = torch.zeros((0, 3))
    projected_points = torch.zeros((0, 2))
    gt_image = torch.zeros((3, 64, 64))
    
    loss = texture_regularization_loss(d_pos, projected_points, gt_image)
    assert loss.item() == 0.0

def test_texture_regularization_loss_weighting():
    """Test that highly textured regions have lower weights and thus lower loss than textureless regions for the same d_pos."""
    H, W = 10, 10
    gt_image = torch.zeros((1, H, W))
    
    # Create a textured region at the top half and textureless at bottom half
    gt_image[:, :5, :] = torch.rand((1, 5, W))
    
    # Gaussian 1 in textured region
    proj1 = torch.tensor([[1.0, 1.0]])
    # Gaussian 2 in textureless region
    proj2 = torch.tensor([[1.0, 8.0]])
    
    d_pos = torch.tensor([[1.0, 1.0, 1.0]])
    
    loss1 = texture_regularization_loss(d_pos, proj1, gt_image, beta=10.0)
    loss2 = texture_regularization_loss(d_pos, proj2, gt_image, beta=10.0)
    
    # Loss 1 (textured) should be smaller than Loss 2 (textureless)
    # because weight = exp(-beta * grad_mag), and grad_mag is higher in textured region.
    assert loss1 < loss2

def test_texture_regularization_loss_out_of_bounds():
    """Test with projected points out of image bounds."""
    H, W = 10, 10
    gt_image = torch.ones((1, H, W))
    
    # Out of bounds projection
    proj = torch.tensor([[15.0, -5.0]])
    d_pos = torch.tensor([[1.0, 1.0, 1.0]])
    
    loss = texture_regularization_loss(d_pos, proj, gt_image)
    # Should not crash, and should use padding_mode="border"
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

def test_texture_regularization_loss_shape():
    """Test with valid shapes."""
    N = 10
    H, W = 64, 64
    d_pos = torch.rand((N, 3))
    projected_points = torch.rand((N, 2)) * 64
    gt_image = torch.rand((3, H, W))
    
    loss = texture_regularization_loss(d_pos, projected_points, gt_image)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
