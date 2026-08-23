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

        norms = d_pos.detach().norm(dim=-1)
        if mask is not None:
            if norms.shape[0] == mask.sum().item():
                self.cum_deformation[mask] += norms
            else:
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
        new_cum_deformation = torch.zeros(num_gaussians, device=self.cum_deformation.device)
        current_n = min(self.cum_deformation.shape[0], num_gaussians)
        new_cum_deformation[:current_n] = self.cum_deformation[:current_n]
        self.cum_deformation = new_cum_deformation

    def reclassify(self, model: 'GaussianModel', threshold: float) -> None:
        """Reclassifies Gaussians based on accumulated deformation threshold.

        Args:
            model (GaussianModel): The GaussianModel instance.
            threshold (float): The threshold for accumulated deformation.
        """
        if self.cum_deformation.shape[0] != model._positions.shape[0]:
            self.resize(model._positions.shape[0])
            
        # Classify as dynamic only if accumulated deformation exceeds threshold
        new_is_dynamic = self.cum_deformation > threshold
        
        # Always keep at least 10% of points dynamic to allow learning
        min_dynamic = min(new_is_dynamic.shape[0], max(1, int(0.1 * new_is_dynamic.shape[0])))
        if new_is_dynamic.sum() < min_dynamic:
            # Keep top K by accumulated deformation
            _, indices = torch.topk(self.cum_deformation, min_dynamic)
            new_is_dynamic = torch.zeros_like(new_is_dynamic)
            new_is_dynamic[indices] = True
            
        # Update model buffer
        model.register_buffer("is_dynamic", new_is_dynamic)
        
        # Reset deformation history
        self.cum_deformation.zero_()
