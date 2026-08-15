"""Densification scheduler for Gaussian Splatting."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict

from sparsechron.models.gaussians import GaussianModel


class DensificationScheduler:
    """Handles cloning, splitting, and pruning of Gaussians."""

    def __init__(
        self,
        model: GaussianModel,
        optimizer: Optimizer,
        max_gaussians: int = 500_000,
        densify_interval: int = 100,
        prune_interval: int = 500,
        grad_threshold: float = 0.00001,
        min_opacity: float = 0.005,
        max_scale: float = 0.5,
    ) -> None:
        """Initializes the densification scheduler.

        Args:
            model: The GaussianModel to modify.
            optimizer: The optimizer to update.
            max_gaussians: Hard cap on number of Gaussians.
            densify_interval: Iterations between densification steps.
            prune_interval: Iterations between pruning steps.
            grad_threshold: Gradient threshold for densification.
            min_opacity: Minimum opacity for pruning.
            max_scale: Maximum scale for pruning.
        """
        self.model = model
        self.optimizer = optimizer
        self.max_gaussians = max_gaussians
        self.densify_interval = densify_interval
        self.prune_interval = prune_interval
        self.grad_threshold = grad_threshold
        self.min_opacity = min_opacity
        self.max_scale = max_scale

        self.device = model.positions.device
        self._init_accumulators()

    def _init_accumulators(self) -> None:
        """Initializes gradient accumulators."""
        n = self.model.positions.shape[0]
        self.accumulated_grads = torch.zeros(n, device=self.device)
        self.denom = torch.zeros(n, device=self.device)

    def step(self, iteration: int) -> None:
        """Performs a scheduling step.

        Args:
            iteration: Current training iteration.
        """
        if self.model._positions.grad is not None:
            # Approximate view-space position gradients using position gradients
            grads = self.model._positions.grad.norm(dim=-1)
            # Ensure shape matches in case of recent modifications
            if grads.shape[0] != self.accumulated_grads.shape[0]:
                self._init_accumulators()
                
            self.accumulated_grads += grads
            self.denom += 1

        if iteration > 0:
            if iteration % self.densify_interval == 0:
                self.densify()
            if iteration % self.prune_interval == 0:
                self.prune()

    def densify(self) -> None:
        """Clones and splits Gaussians based on accumulated gradients."""
        n_init = self.model.positions.shape[0]
        if n_init >= self.max_gaussians:
            self._init_accumulators()
            return

        avg_grads = self.accumulated_grads / self.denom.clamp(min=1)
        mask = avg_grads >= self.grad_threshold
        
        n_candidates = mask.sum().item()
        if n_candidates == 0:
            self._init_accumulators()
            return
            
        n_allowed = self.max_gaussians - n_init
        if n_candidates > n_allowed:
            top_k_indices = torch.topk(avg_grads[mask], n_allowed).indices
            new_mask = torch.zeros_like(mask)
            valid_indices = torch.nonzero(mask).squeeze(1)[top_k_indices]
            new_mask[valid_indices] = True
            mask = new_mask
            
        self._clone_gaussians(mask)
        self._init_accumulators()

    def prune(self) -> None:
        """Prunes Gaussians based on opacity and scale."""
        opacities = torch.sigmoid(self.model._opacities).squeeze(-1)
        scales = torch.exp(self.model._scales)
        max_scales = scales.max(dim=-1).values

        prune_mask = (opacities < self.min_opacity) | (max_scales > self.max_scale)
        keep_mask = ~prune_mask
        
        if not prune_mask.any():
            return
            
        if not keep_mask.any():
            # Don't prune all Gaussians
            return

        self._filter_gaussians(keep_mask)
        self._init_accumulators()

    def _clone_gaussians(self, mask: torch.Tensor) -> None:
        """Clones Gaussians indicated by the mask.

        Args:
            mask: Boolean tensor indicating which Gaussians to clone.
        """
        if not mask.any():
            return

        new_params = {}
        for name in ["_positions", "_scales", "_rotations", "_opacities", "_sh_coeffs"]:
            param = getattr(self.model, name)
            if name == "_positions":
                # Jitter positions slightly so they don't perfectly overlap
                noise = torch.randn_like(param.data[mask]) * 0.001
                new_data = torch.cat([param.data, param.data[mask] + noise], dim=0)
            else:
                new_data = torch.cat([param.data, param.data[mask]], dim=0)
            new_params[name] = new_data
            
        self._update_parameters_and_optimizer(new_params, mode="cat", mask=mask)
        
        if hasattr(self.model, "is_dynamic"):
            new_is_dynamic = torch.cat([self.model.is_dynamic, self.model.is_dynamic[mask]], dim=0)
            self.model.register_buffer("is_dynamic", new_is_dynamic)

    def _filter_gaussians(self, keep_mask: torch.Tensor) -> None:
        """Filters Gaussians keeping only those indicated by the mask.

        Args:
            keep_mask: Boolean tensor indicating which Gaussians to keep.
        """
        new_params = {}
        for name in ["_positions", "_scales", "_rotations", "_opacities", "_sh_coeffs"]:
            param = getattr(self.model, name)
            new_data = param.data[keep_mask]
            new_params[name] = new_data
            
        self._update_parameters_and_optimizer(new_params, mode="filter", mask=keep_mask)
        
        if hasattr(self.model, "is_dynamic"):
            new_is_dynamic = self.model.is_dynamic[keep_mask]
            self.model.register_buffer("is_dynamic", new_is_dynamic)

    def _update_parameters_and_optimizer(
        self, new_params: Dict[str, torch.Tensor], mode: str, mask: torch.Tensor
    ) -> None:
        """Updates model parameters and optimizer state.

        Args:
            new_params: Dictionary mapping parameter names to new tensors.
            mode: "cat" for cloning, "filter" for pruning.
            mask: The mask used for the operation.
        """
        for name, new_tensor in new_params.items():
            old_param = getattr(self.model, name)
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            setattr(self.model, name, new_param)

            # Update optimizer
            for group in self.optimizer.param_groups:
                for i, p in enumerate(group["params"]):
                    if p is old_param:
                        group["params"][i] = new_param
                        
                        if old_param in self.optimizer.state:
                            old_state = self.optimizer.state.pop(old_param)
                            new_state = {}
                            for k, v in old_state.items():
                                if k == "step":
                                    new_state[k] = v
                                elif torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == old_param.shape[0]:
                                    if mode == "cat":
                                        new_state[k] = torch.cat([v, v[mask]], dim=0)
                                    else:
                                        new_state[k] = v[mask]
                                else:
                                    new_state[k] = v
                            self.optimizer.state[new_param] = new_state
