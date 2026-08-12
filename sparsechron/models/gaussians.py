import torch
from torch import nn
from typing import Dict
from sparsechron.models.deformation import apply_offsets


class GaussianModel(nn.Module):
    """
    A PyTorch module representing a set of 3D Gaussians.

    Attributes:
        _positions (nn.Parameter): The positions of the Gaussians (N, 3).
        _scales (nn.Parameter): The unactivated scales of the Gaussians (N, 3).
        _rotations (nn.Parameter): The unactivated rotations (quaternions) (N, 4).
        _opacities (nn.Parameter): The unactivated opacities of the Gaussians (N, 1).
        _sh_coeffs (nn.Parameter): The spherical harmonics coefficients (N, D, 3).
    """

    def __init__(self, initial_values: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the GaussianModel with given values.

        Args:
            initial_values: A dictionary containing the initial values for the
                Gaussians. Expected keys are 'positions', 'scales', 'rotations',
                'opacities', and 'sh_coeffs'.
        """
        super().__init__()
        required_keys = ["positions", "scales", "rotations", "opacities", "sh_coeffs"]
        for key in required_keys:
            if key not in initial_values:
                raise ValueError(f"Missing required key '{key}' in initial_values.")

        self._positions = nn.Parameter(initial_values["positions"])
        self._scales = nn.Parameter(initial_values["scales"])
        self._rotations = nn.Parameter(initial_values["rotations"])
        self._opacities = nn.Parameter(initial_values["opacities"])
        self._sh_coeffs = nn.Parameter(initial_values["sh_coeffs"])
        
        self.register_buffer(
            "is_dynamic",
            torch.ones(self._positions.shape[0], dtype=torch.bool, device=self._positions.device)
        )

    @property
    def positions(self) -> torch.Tensor:
        """
        Gets the activated positions of the Gaussians.

        Returns:
            The positions (N, 3).
        """
        return self._positions

    @property
    def scales(self) -> torch.Tensor:
        """
        Gets the activated scales of the Gaussians.

        Returns:
            The exponentially activated scales (N, 3).
        """
        return torch.exp(self._scales)

    @property
    def rotations(self) -> torch.Tensor:
        """
        Gets the activated rotations of the Gaussians.

        Returns:
            The normalized quaternions (N, 4).
        """
        return torch.nn.functional.normalize(self._rotations, p=2.0, dim=-1)

    @property
    def opacities(self) -> torch.Tensor:
        """
        Gets the activated opacities of the Gaussians.

        Returns:
            The sigmoid activated opacities (N, 1).
        """
        return torch.sigmoid(self._opacities)

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """
        Gets the spherical harmonics coefficients.

        Returns:
            The spherical harmonics coefficients (N, D, 3).
        """
        return self._sh_coeffs

    def get_deformed(self, deformation_mlp: nn.Module, timestep: float) -> Dict[str, torch.Tensor]:
        """Gets the fully activated and deformed parameters for a specific timestep.

        Args:
            deformation_mlp (nn.Module): The deformation MLP.
            timestep (float): The current timestep.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the deformed and
                activated parameters ('positions', 'scales', 'rotations',
                'opacities', 'sh_coeffs').
        """

        
        # Get activated base parameters
        pos = self.positions
        rot = self.rotations
        scale = self.scales
        opacity = self.opacities
        sh = self.sh_coeffs

        # Ensure is_dynamic mask perfectly matches the current number of Gaussians
        # (in case the scheduler updated Gaussians but the buffer update lagged)
        current_n = pos.shape[0]
        if self.is_dynamic.shape[0] != current_n:
            if self.is_dynamic.shape[0] > current_n:
                is_dynamic = self.is_dynamic[:current_n]
            else:
                padding = torch.ones(current_n - self.is_dynamic.shape[0], dtype=torch.bool, device=pos.device)
                is_dynamic = torch.cat([self.is_dynamic, padding])
        else:
            is_dynamic = self.is_dynamic

        if not is_dynamic.any():
            return {
                "positions": pos,
                "scales": scale,
                "rotations": rot,
                "opacities": opacity,
                "sh_coeffs": sh,
                "d_pos": torch.zeros_like(pos),
            }

        # Filter dynamic parameters
        dyn_pos = pos[is_dynamic]
        
        # We need to apply offsets. Wait, the prompt says:
        # "Apply offsets to get deformed dynamic parameters."
        # Should we apply them to activated or unactivated? 
        # Typically offsets are applied to unactivated scale/rot, or maybe activated?
        # "New pos = pos + d_pos, new scale = scale + d_scale. For rotations, use quaternion multiplication quat_mul(rot, d_rot) and then normalize."
        # If we use `pos`, `rot`, `scale`, let's just use the activated ones since `apply_offsets` uses `quat_mul` and `new_scale = scale + d_scale`.
        
        dyn_rot_act = rot[is_dynamic]
        dyn_scale_act = scale[is_dynamic]

        # Prepare inputs for MLP
        times = torch.full((dyn_pos.shape[0], 1), timestep, device=dyn_pos.device, dtype=dyn_pos.dtype)
        
        # Get offsets from MLP
        d_pos, d_rot, d_scale = deformation_mlp(dyn_pos, times)
        
        # Apply offsets
        def_pos, def_rot, def_scale = apply_offsets(
            dyn_pos, dyn_rot_act, dyn_scale_act, d_pos, d_rot, d_scale
        )

        # Re-combine with static parameters
        final_pos = pos.clone()
        final_rot = rot.clone()
        final_scale = scale.clone()

        final_pos[is_dynamic] = def_pos
        final_rot[is_dynamic] = def_rot
        final_scale[is_dynamic] = def_scale

        d_pos_full = torch.zeros_like(pos)
        d_pos_full[is_dynamic] = d_pos

        return {
            "positions": final_pos,
            "scales": final_scale,
            "rotations": final_rot,
            "opacities": opacity,
            "sh_coeffs": sh,
            "d_pos": d_pos_full,
        }
