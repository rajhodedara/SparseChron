import torch
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.utils.camera import Camera

def test_gaussian_renderer() -> None:
    num_pts = 100
    dim_sh = 1
    device = "cpu"
    
    initial_values = {
        "positions": torch.rand((num_pts, 3), device=device),
        "scales": torch.rand((num_pts, 3), device=device),
        "rotations": torch.rand((num_pts, 4), device=device),
        "opacities": torch.rand((num_pts, 1), device=device),
        "sh_coeffs": torch.rand((num_pts, dim_sh, 3), device=device),
    }
    model = GaussianModel(initial_values)
    
    camera = Camera(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        R=torch.eye(3, device=device),
        T=torch.zeros(3, device=device)
    )
    
    renderer = GaussianRenderer()
    
    output = renderer.render(model, camera)
    
    assert "rgb" in output
    assert "depth" in output
    
    rgb = output["rgb"]
    depth = output["depth"]
    
    assert rgb.shape == (camera.height, camera.width, 3)
    assert depth.shape == (camera.height, camera.width, 1)
