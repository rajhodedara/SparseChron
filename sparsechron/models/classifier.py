import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sparsechron.models.gaussians import GaussianModel


class StaticDynamicClassifier:
    """Classifies Gaussians as static or dynamic based on accumulated deformation."""

    def __init__(self, num_gaussians: int) -> None:
        """Initializes the classifier.

        Args:
            num_gaussians (int): The number of Gaussians to track.
        """
        self.cum_deformation = torch.zeros(num_gaussians)

    def update(self, d_pos: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Updates the accumulated deformation.

        Args:
            d_pos (torch.Tensor): Position offsets of Gaussians.
            mask (torch.Tensor, optional): Boolean mask of which Gaussians to update.
        """
        if self.cum_deformation.device != d_pos.device:
            self.cum_deformation = self.cum_deformation.to(d_pos.device)

        norms = d_pos.norm(dim=-1)
        if mask is not None:
            self.cum_deformation[mask] += norms[mask]
        elif norms.shape[0] == self.cum_deformation.shape[0]:
            self.cum_deformation += norms
        else:
            raise ValueError("d_pos must match the number of Gaussians or provide a mask.")

    def resize(self, num_gaussians: int) -> None:
        """Re-allocates the accumulated deformation tensor.

        Args:
            num_gaussians (int): The new number of Gaussians.
        """
        self.cum_deformation = torch.zeros(num_gaussians, device=self.cum_deformation.device)

    def reclassify(self, model: 'GaussianModel', threshold: float) -> None:
        """Reclassifies Gaussians based on accumulated deformation threshold.

        Args:
            model (GaussianModel): The GaussianModel instance.
            threshold (float): The threshold for accumulated deformation.
        """
        # Set to False if cum_deformation < threshold
        is_static = self.cum_deformation.to(model.is_dynamic.device) < threshold
        model.is_dynamic[is_static] = False
        
        # Reset the buffer
        self.cum_deformation.zero_()
