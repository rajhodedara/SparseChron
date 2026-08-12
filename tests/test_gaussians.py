import torch
from sparsechron.models.gaussians import GaussianModel

def test_gaussian_model_initialization() -> None:
    num_pts = 10
    dim_sh = 16
    initial_values = {
        "positions": torch.rand((num_pts, 3)),
        "scales": torch.rand((num_pts, 3)),
        "rotations": torch.rand((num_pts, 4)),
        "opacities": torch.rand((num_pts, 1)),
        "sh_coeffs": torch.rand((num_pts, dim_sh, 3)),
    }
    
    model = GaussianModel(initial_values)
    
    assert model.positions.shape == (num_pts, 3)
    assert model.scales.shape == (num_pts, 3)
    assert model.rotations.shape == (num_pts, 4)
    assert model.opacities.shape == (num_pts, 1)
    assert model.sh_coeffs.shape == (num_pts, dim_sh, 3)

def test_gaussian_model_activations() -> None:
    num_pts = 5
    dim_sh = 16
    
    # Pre-activation values
    scales_raw = torch.tensor([[0.0, 1.0, -1.0]] * num_pts)
    rotations_raw = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * num_pts)
    opacities_raw = torch.tensor([[0.0]] * num_pts) # sigmoid(0) = 0.5
    
    initial_values = {
        "positions": torch.zeros((num_pts, 3)),
        "scales": scales_raw,
        "rotations": rotations_raw,
        "opacities": opacities_raw,
        "sh_coeffs": torch.zeros((num_pts, dim_sh, 3)),
    }
    
    model = GaussianModel(initial_values)
    
    # Check scales (exponential)
    scales_act = model.scales
    assert torch.all(scales_act > 0)
    assert torch.allclose(scales_act[0], torch.tensor([1.0, torch.exp(torch.tensor(1.0)), torch.exp(torch.tensor(-1.0))]))
    
    # Check rotations (normalization)
    rotations_act = model.rotations
    norms = torch.norm(rotations_act, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))
    
    # Check opacities (sigmoid)
    opacities_act = model.opacities
    assert torch.all((opacities_act > 0.0) & (opacities_act < 1.0))
    assert torch.allclose(opacities_act, torch.full((num_pts, 1), 0.5))
