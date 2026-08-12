import pytest
import torch
from sparsechron.utils.camera import Camera


def test_camera_initialization():
    R = torch.eye(3)
    T = torch.zeros(3)
    
    # Valid camera
    cam = Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=T)
    assert cam.fx == 100.0

    # Fail fast checks
    with pytest.raises(ValueError, match="R must be shape"):
        Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=torch.eye(4), T=T)

    with pytest.raises(ValueError, match="T must be shape"):
        Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=torch.zeros(4))

    with pytest.raises(ValueError, match="Focal lengths must be positive"):
        Camera(fx=-100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=T)

    with pytest.raises(ValueError, match="Dimensions must be positive"):
        Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=-100, height=100, R=R, T=T)


def test_camera_world_to_camera_shape():
    R = torch.eye(3)
    T = torch.zeros(3)
    cam = Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=T)
    
    pts = torch.randn(10, 3)
    pts_cam = cam.world_to_camera(pts)
    assert pts_cam.shape == (10, 3)

    with pytest.raises(ValueError):
        cam.world_to_camera(torch.randn(10, 2))


def test_camera_project_shape():
    R = torch.eye(3)
    T = torch.zeros(3)
    cam = Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=T)
    
    pts = torch.randn(10, 3)
    pixels = cam.project(pts)
    assert pixels.shape == (10, 2)

    with pytest.raises(ValueError):
        cam.project(torch.randn(10, 2))


def test_camera_unproject_project_round_trip():
    torch.manual_seed(42)
    # Give a non-trivial pose
    # Let's rotate a bit
    theta = torch.tensor(0.1)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta),  torch.cos(theta), 0.0],
        [0.0,               0.0,              1.0]
    ])
    T = torch.tensor([0.1, -0.2, 0.5])
    
    cam = Camera(fx=200.0, fy=200.0, cx=150.0, cy=100.0, width=300, height=200, R=R, T=T)
    
    # Generate random depth map with positive depth values
    depth_map = torch.rand(200, 300) * 5.0 + 1.0  # depths between 1.0 and 6.0
    
    # Unproject
    pts_world = cam.unproject(depth_map)
    assert pts_world.shape == (200 * 300, 3)
    
    # Project back
    pixels = cam.project(pts_world)
    assert pixels.shape == (200 * 300, 2)
    
    # The pixels should perfectly match the grid!
    # Because unproject creates points based on a regular grid of (v, u) -> (y, x).
    # Wait, the unproject creates grid using v_grid, u_grid from 0 to h-1, 0 to w-1.
    v_grid, u_grid = torch.meshgrid(
        torch.arange(200, dtype=depth_map.dtype),
        torch.arange(300, dtype=depth_map.dtype),
        indexing='ij'
    )
    expected_u = u_grid.flatten()
    expected_v = v_grid.flatten()
    expected_pixels = torch.stack([expected_u, expected_v], dim=-1)
    
    assert torch.allclose(pixels, expected_pixels, atol=1e-4)

    with pytest.raises(ValueError):
        cam.unproject(torch.randn(100, 100, 1))
    
    with pytest.raises(ValueError):
        cam.unproject(torch.randn(100, 100)) # wrong shape
