import torch
from torch import nn
from typing import Tuple

from sparsechron.utils.transforms import quaternion_multiply as quat_mul
from sparsechron.utils.transforms import normalize_quaternion


def apply_offsets(
    pos: torch.Tensor,
    rot: torch.Tensor,
    scale: torch.Tensor,
    d_pos: torch.Tensor,
    d_rot: torch.Tensor,
    d_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies offsets to positions, rotations, and scales.

    Args:
        pos (torch.Tensor): Original positions (N, 3).
        rot (torch.Tensor): Original rotations (N, 4).
        scale (torch.Tensor): Original scales (N, 3).
        d_pos (torch.Tensor): Position offsets (N, 3).
        d_rot (torch.Tensor): Rotation offsets (N, 4).
        d_scale (torch.Tensor): Scale offsets (N, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Deformed pos, rot, scale.
    """
    new_pos = pos + d_pos
    new_scale = scale * torch.exp(d_scale)
    new_rot = normalize_quaternion(quat_mul(rot, d_rot))
    return new_pos, new_rot, new_scale


class DeformationMLP(nn.Module):
    """MLP for computing temporal deformations of 3D Gaussians."""

    def __init__(self, spatial_freqs: int = 6, time_freqs: int = 6) -> None:
        """Initializes the DeformationMLP.

        Args:
            spatial_freqs (int): Number of frequencies for spatial encoding.
            time_freqs (int): Number of frequencies for time encoding.
        """
        super().__init__()
        self.spatial_freqs = spatial_freqs
        self.time_freqs = time_freqs

        in_channels = 3 + 3 * 2 * spatial_freqs + 1 + 1 * 2 * time_freqs
        hidden_channels = 128

        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        for _ in range(4):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
        final_layer = nn.Linear(hidden_channels, 10)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        final_layer.bias.data[0] = 1.0  # Identity quaternion real part (w, x, y, z format)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

    def positional_encoding(self, x: torch.Tensor, num_freqs: int) -> torch.Tensor:
        """Computes positional encoding for the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            num_freqs (int): Number of frequencies.

        Returns:
            torch.Tensor: Positionally encoded tensor.
        """
        encodings = [x]
        for i in range(num_freqs):
            encodings.append(torch.sin((2.0**i) * x))
            encodings.append(torch.cos((2.0**i) * x))
        return torch.cat(encodings, dim=-1)

    def forward(
        self, positions: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes deformations for given positions and times.

        Args:
            positions (torch.Tensor): Spatial coordinates (N, 3).
            times (torch.Tensor): Time coordinates (N, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: d_pos, d_rot, d_scale.
        """
        pos_enc = self.positional_encoding(positions, self.spatial_freqs)
        time_enc = self.positional_encoding(times, self.time_freqs)

        x = torch.cat([pos_enc, time_enc], dim=-1)
        output = self.mlp(x)

        d_pos = output[..., :3]
        d_rot = output[..., 3:7]
        d_scale = output[..., 7:10]

        return d_pos, d_rot, d_scale
