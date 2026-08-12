import torch
from torch import nn
from typing import Dict


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
