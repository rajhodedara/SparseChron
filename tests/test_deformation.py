import torch
from sparsechron.models.deformation import apply_offsets, DeformationMLP

def test_apply_offsets():
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scale = torch.tensor([[2.0, 2.0, 2.0]])
    
    d_pos = torch.tensor([[0.5, -0.5, 1.0]])
    d_rot = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # not normalized, but normalize_quaternion handles it
    d_scale = torch.tensor([[0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(0.5))]])
    
    new_pos, new_rot, new_scale = apply_offsets(pos, rot, scale, d_pos, d_rot, d_scale)
    
    assert torch.allclose(new_pos, torch.tensor([[1.5, 1.5, 4.0]]))
    assert torch.allclose(new_scale, torch.tensor([[2.0, 4.0, 1.0]]))
    
    # Check quaternion normalization
    assert torch.allclose(torch.norm(new_rot, dim=-1), torch.ones(1))
    
def test_deformation_mlp_initialization():
    mlp = DeformationMLP(spatial_freqs=2, time_freqs=2)
    
    # Should output zeros for d_pos and d_scale, and identity [1,0,0,0] for d_rot
    pos = torch.zeros((1, 3))
    time = torch.zeros((1, 1))
    
    d_pos, d_rot, d_scale = mlp(pos, time)
    
    assert torch.allclose(d_pos, torch.zeros((1, 3)), atol=1e-6)
    assert torch.allclose(d_rot, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(d_scale, torch.zeros((1, 3)), atol=1e-6)

def test_deformation_mlp_gradient_flow():
    mlp = DeformationMLP()
    pos = torch.rand((5, 3), requires_grad=True)
    time = torch.rand((5, 1), requires_grad=True)
    
    d_pos, d_rot, d_scale = mlp(pos, time)
    loss = d_pos.sum() + d_rot.sum() + d_scale.sum()
    loss.backward()
    
    assert pos.grad is not None
    assert time.grad is not None
    
    for param in mlp.parameters():
        assert param.grad is not None
