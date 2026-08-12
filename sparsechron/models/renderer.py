import torch
from typing import Dict, Any

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

    def render(self, model: GaussianModel, camera: Camera) -> Dict[str, torch.Tensor]:
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

        # Get activated properties
        means = model.positions.contiguous()
        quats = model.rotations.contiguous()
        scales = model.scales.contiguous()
        opacities = model.opacities.squeeze(-1).contiguous()

        # Extract 0-th order SH for base color
        sh_c0 = 0.28209479177387814
        colors = model.sh_coeffs[:, 0, :] * sh_c0 + 0.5
        colors = torch.clamp(colors, 0.0, 1.0).contiguous()

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
            depth = meta.get("depths", torch.zeros_like(rgb[..., :1]))
        else:
            rgb = out[0] if isinstance(out, tuple) else out
            depth = torch.zeros(
                (camera.height, camera.width, 1), device=device
            )

        # Squeeze camera dimension (C=1)
        if rgb.ndim == 4:
            rgb = rgb.squeeze(0)
        if depth.ndim == 4:
            depth = depth.squeeze(0)

        return {
            "rgb": rgb,
            "depth": depth
        }
