import torch
from typing import Dict, Any, Optional

from sparsechron.models.gaussians import GaussianModel
from sparsechron.utils.camera import Camera

try:
    from gsplat import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False


class GaussianRenderer:
    """Renderer for 3D Gaussians using gsplat."""

    def __init__(self) -> None:
        """Initializes the GaussianRenderer."""
        pass

    def render(
        self,
        model: GaussianModel,
        camera: Camera,
        deformed_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Renders the given Gaussian model from the perspective of the camera.

        Args:
            model: The GaussianModel to render.
            camera: The Camera to render from.

        Returns:
            A dictionary containing 'rgb' (H, W, 3) and 'depth' (H, W, 1) tensors.
        """
        device = model.positions.device

        if not HAS_GSPLAT:
            return {
                "rgb": torch.zeros((camera.height, camera.width, 3), device=device),
                "depth": torch.zeros((camera.height, camera.width, 1), device=device)
            }

        if deformed_params is not None:
            means = deformed_params["positions"].contiguous()
            quats = deformed_params["rotations"].contiguous()
            scales = deformed_params["scales"].contiguous()
            opacities = deformed_params["opacities"].squeeze(-1).contiguous()
            sh_coeffs = deformed_params["sh_coeffs"]
        else:
            means = model.positions.contiguous()
            quats = model.rotations.contiguous()
            scales = model.scales.contiguous()
            opacities = model.opacities.squeeze(-1).contiguous()
            sh_coeffs = model.sh_coeffs


        # Use a smooth sigmoid activation to prevent dead gradients
        # If we have higher degree SH, evaluate them using view directions
        if sh_coeffs.shape[1] > 1:
            from gsplat import spherical_harmonics
            import math
            
            # Extract camera center in world space: C = -R^T * T
            if hasattr(camera, 'R') and hasattr(camera, 'T'):
                cam_center = -(camera.R.to(device).T @ camera.T.to(device).squeeze())
            elif hasattr(camera, 'w2c'):
                w2c = camera.w2c.to(device)
                cam_center = -(w2c[:3, :3].T @ w2c[:3, 3])
            else:
                cam_center = torch.zeros(3, device=device)
                
            dirs = means - cam_center
            dirs = torch.nn.functional.normalize(dirs, dim=-1)
            degrees = int(math.sqrt(sh_coeffs.shape[1])) - 1
            
            sh_eval = spherical_harmonics(degrees, dirs, sh_coeffs)
            colors = torch.sigmoid(sh_eval).contiguous()
        else:
            colors = torch.sigmoid(sh_coeffs[:, 0, :]).contiguous()

        # Build ks (intrinsics)
        k = torch.eye(3, dtype=torch.float32, device=device)
        k[0, 0] = camera.fx
        k[1, 1] = camera.fy
        k[0, 2] = camera.cx
        k[1, 2] = camera.cy
        ks = k.unsqueeze(0).contiguous()  # (1, 3, 3)

        # Build viewmat (world to camera)
        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        if hasattr(camera, 'w2c'):
            viewmat = camera.w2c.to(device)
        elif hasattr(camera, 'R') and hasattr(camera, 'T'):
            viewmat[:3, :3] = camera.R.to(device)
            viewmat[:3, 3] = camera.T.to(device).squeeze()
        viewmats = viewmat.unsqueeze(0).contiguous()  # (1, 4, 4)

        # Call gsplat rasterization
        out = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=ks,
            width=camera.width,
            height=camera.height,
            render_mode="RGB+ED",
        )

        # Handle different potential return formats from gsplat 1.0.0
        if isinstance(out, tuple) and len(out) >= 3:
            rgb = out[0]
            meta = out[2]
            if "gaussian_ids" in meta and "means2d" in meta:
                nnz_means2d = meta["means2d"]
                gaussian_ids = meta["gaussian_ids"]
                means2d = torch.zeros((means.shape[0], 2), device=device)
                means2d[gaussian_ids] = nnz_means2d
            else:
                means2d = meta.get("means2d", torch.zeros_like(means[..., :2]))
                if means2d.ndim == 3:
                    means2d = means2d.squeeze(0)
        else:
            rgb = out[0] if isinstance(out, tuple) else out
            means2d = torch.zeros_like(means[..., :2])

        # If render_mode="RGB+ED", rgb will have 4 channels.
        if rgb.shape[-1] == 4:
            depth = rgb[..., 3:4]
            rgb = rgb[..., :3]
        else:
            depth = torch.zeros((rgb.shape[0] if rgb.ndim == 4 else 1, camera.height, camera.width, 1), device=device)
            if rgb.ndim == 3:
                depth = depth.squeeze(0)

        # Squeeze camera dimension (C=1) if batched
        if rgb.ndim == 4:
            rgb = rgb.squeeze(0)
        if depth.ndim == 4:
            depth = depth.squeeze(0)

        return {
            "rgb": rgb,
            "depth": depth,
            "means2d": means2d
        }
