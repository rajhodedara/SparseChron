"""Camera geometry utilities."""

from dataclasses import dataclass
import torch


@dataclass
class Camera:
    """Dataclass to manage camera intrinsics and extrinsics.
    
    Attributes:
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        R (torch.Tensor): 3x3 rotation matrix from world to camera.
        T (torch.Tensor): 3-element translation vector from world to camera.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: torch.Tensor
    T: torch.Tensor
    
    def __post_init__(self) -> None:
        if self.R.shape != (3, 3):
            raise ValueError(f"R must be shape (3, 3), got {self.R.shape}")
        if self.T.shape not in [(3,), (3, 1)]:
            raise ValueError(f"T must be shape (3,) or (3, 1), got {self.T.shape}")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.width}x{self.height}")
    
    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        """Transforms 3D world points to 3D camera coordinates.
        
        Args:
            points (torch.Tensor): Tensor of shape (N, 3) representing 3D world points.
            
        Returns:
            torch.Tensor: Tensor of shape (N, 3) representing points in camera coordinates.
        """
        if points.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {points.shape}")
        # R transforms column vectors, so for (N, 3) row vectors:
        # X_cam^T = X_world^T R^T + T^T
        return points @ self.R.T + self.T
        
    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Transforms 3D world points to 2D image coordinates (pixels).
        
        Args:
            points (torch.Tensor): Tensor of shape (N, 3) representing 3D world points.
            
        Returns:
            torch.Tensor: Tensor of shape (N, 2) representing 2D image coordinates.
        """
        if points.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {points.shape}")
        points_cam = self.world_to_camera(points)
        x = points_cam[..., 0]
        y = points_cam[..., 1]
        z = points_cam[..., 2]
        
        # Avoid division by zero
        z_safe = torch.where(z.abs() < 1e-6, torch.copysign(torch.tensor(1e-6, device=z.device, dtype=z.dtype), z), z)
        
        u = (x * self.fx / z_safe) + self.cx
        v = (y * self.fy / z_safe) + self.cy
        
        return torch.stack([u, v], dim=-1)
        
    def unproject(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Converts a 2D depth map to a 3D point cloud in world space.
        
        Args:
            depth_map (torch.Tensor): Tensor of shape (H, W) representing depth values.
            
        Returns:
            torch.Tensor: Tensor of shape (H*W, 3) representing 3D world points.
        """
        if depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D, got {depth_map.ndim}D")
        h, w = depth_map.shape
        if h != self.height or w != self.width:
            raise ValueError(f"depth_map shape {depth_map.shape} does not match camera {self.height}x{self.width}")
            
        device = depth_map.device
        
        # Create pixel grid
        v_grid, u_grid = torch.meshgrid(
            torch.arange(h, device=device, dtype=depth_map.dtype),
            torch.arange(w, device=device, dtype=depth_map.dtype),
            indexing='ij'
        )
        
        u = u_grid.flatten()
        v = v_grid.flatten()
        z = depth_map.flatten()
        
        # Convert to camera coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points_cam = torch.stack([x, y, z], dim=-1)  # (H*W, 3)
        
        # Convert to world coordinates
        # X_cam = R X_world + T
        # X_world = R^T (X_cam - T)
        # For row vectors: X_world^T = (X_cam^T - T^T) R
        points_world = (points_cam - self.T) @ self.R
        
        return points_world
