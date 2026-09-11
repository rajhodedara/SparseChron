"""Densification scheduler for Gaussian Splatting.

Implements 3DGS-style adaptive density control: Gaussians with high
positional gradients are either SPLIT (if oversized) or CLONED (if small),
and transparent/oversized Gaussians are pruned.

Densification rebuilds all Gaussian tensors in a single pass with an explicit
row mapping, so parameters, optimizer state (Adam moments), and the
``is_dynamic`` buffer always stay perfectly aligned.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Optional

from sparsechron.models.gaussians import GaussianModel
from sparsechron.utils.transforms import quaternion_to_rotation_matrix

_PARAM_NAMES = ["_positions", "_scales", "_rotations", "_opacities", "_sh_coeffs"]


class DensificationScheduler:
    """Handles cloning, splitting, and pruning of Gaussians."""

    def __init__(
        self,
        model: GaussianModel,
        optimizer: Optimizer,
        max_gaussians: int = 500_000,
        densify_interval: int = 100,
        prune_interval: int = 500,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_scale: float = 0.5,
        split_scale_threshold: float = 0.05,
        clone_noise: float = 0.001,
        split_samples: int = 2,
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
            split_scale_threshold: Activated max-scale above which a Gaussian is
                split instead of cloned. Interpreted in scene units (use with a
                normalized scene, e.g. [-1, 1]).
            clone_noise: Positional jitter for cloned Gaussians.
            split_samples: Number of children produced per split.
        """
        self.model = model
        self.optimizer = optimizer
        self.max_gaussians = max_gaussians
        self.densify_interval = densify_interval
        self.prune_interval = prune_interval
        self.grad_threshold = grad_threshold
        self.min_opacity = min_opacity
        self.max_scale = max_scale
        self.split_scale_threshold = split_scale_threshold
        self.clone_noise = clone_noise
        self.split_samples = split_samples

        self.device = model.positions.device
        self._init_accumulators()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
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
        """Clones and splits Gaussians based on accumulated gradients.

        Oversized Gaussians (max activated scale > split_scale_threshold) are
        split into ``split_samples`` children each; the rest are cloned in
        place. Both operations are applied in one rebuild so the parent row
        mapping stays consistent.
        """
        n_init = self.model.positions.shape[0]
        if n_init >= self.max_gaussians:
            self._init_accumulators()
            return

        avg_grads = self.accumulated_grads / self.denom.clamp(min=1)
        mask = avg_grads >= self.grad_threshold

        scales_max = torch.exp(self.model._scales).max(dim=-1).values
        split_sel = mask & (scales_max > self.split_scale_threshold)
        clone_sel = mask & ~split_sel

        n_clone = int(clone_sel.sum().item())
        n_split = int(split_sel.sum().item())
        if n_clone == 0 and n_split == 0:
            self._init_accumulators()
            return

        # Respect the hard Gaussian cap: thin the candidate sets proportionally.
        # Net growth = one row per split (parent -> children) + one per clone.
        budget = self.max_gaussians - n_init
        total_new = n_clone + n_split
        if total_new > budget:
            if budget <= 0:
                self._init_accumulators()
                return
            frac = budget / total_new
            clone_sel = self._thin_mask(clone_sel, int(n_clone * frac))
            split_sel = self._thin_mask(split_sel, int(n_split * frac))
            n_clone = int(clone_sel.sum().item())
            n_split = int(split_sel.sum().item())
            if n_clone == 0 and n_split == 0:
                self._init_accumulators()
                return

        ar = torch.arange(n_init, device=self.device)
        surv_idx = ar[~split_sel & ~clone_sel]
        clone_idx = ar[clone_sel]
        split_idx = ar[split_sel]

        # --- Clone children: exact copies with positional jitter ---
        clone_pos = self.model._positions.data[clone_idx] + (
            torch.randn(
                (n_clone, 3),
                device=self.device,
                dtype=self.model._positions.dtype,
            )
            * self.clone_noise
        )

        # --- Split children: sampled inside the parent ellipsoid, 1.6x smaller ---
        child_pos, child_scales_raw, child_rot_raw, child_opac_raw, child_sh = (
            self._sample_split_children(split_idx)
        )

        new_tensors = {
            "_positions": torch.cat(
                [self.model._positions.data[surv_idx], clone_pos, child_pos], dim=0
            ),
            "_scales": torch.cat(
                [
                    self.model._scales.data[surv_idx],
                    self.model._scales.data[clone_idx],
                    child_scales_raw,
                ],
                dim=0,
            ),
            "_rotations": torch.cat(
                [
                    self.model._rotations.data[surv_idx],
                    self.model._rotations.data[clone_idx],
                    child_rot_raw,
                ],
                dim=0,
            ),
            "_opacities": torch.cat(
                [
                    self.model._opacities.data[surv_idx],
                    self.model._opacities.data[clone_idx],
                    child_opac_raw,
                ],
                dim=0,
            ),
            "_sh_coeffs": torch.cat(
                [
                    self.model._sh_coeffs.data[surv_idx],
                    self.model._sh_coeffs.data[clone_idx],
                    child_sh,
                ],
                dim=0,
            ),
        }

        # New row i descends from old row row_source[i]; split parents are
        # dropped (their children replace them).
        row_source = torch.cat(
            [surv_idx, clone_idx, split_idx.repeat_interleave(self.split_samples)],
            dim=0,
        )
        self._apply_gaussians(new_tensors, row_source)
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

        surv_idx = torch.nonzero(keep_mask).squeeze(1)
        new_tensors = {
            name: getattr(self.model, name).data[surv_idx] for name in _PARAM_NAMES
        }
        self._apply_gaussians(new_tensors, surv_idx)
        self._init_accumulators()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sample_split_children(self, split_idx: torch.Tensor):
        """Samples split children for the given parent indices.

        Children are offset inside the parent's ellipsoid (rotated into world
        space) and given scales reduced by 1.6x, following 3DGS.
        """
        M = int(split_idx.numel())
        if M == 0:
            # Indexing with an empty index tensor preserves trailing dims.
            return (
                self.model._positions.data[split_idx],
                self.model._scales.data[split_idx],
                self.model._rotations.data[split_idx],
                self.model._opacities.data[split_idx],
                self.model._sh_coeffs.data[split_idx],
        )

        samples = self.split_samples
        parent_pos = self.model._positions.data[split_idx]         # (M, 3)
        parent_scales_raw = self.model._scales.data[split_idx]     # (M, 3) log-space
        parent_rot_raw = self.model._rotations.data[split_idx]     # (M, 4)
        parent_opac_raw = self.model._opacities.data[split_idx]    # (M, 1)
        parent_sh = self.model._sh_coeffs.data[split_idx]          # (M, D, 3)

        total = M * samples
        scales_act = torch.exp(parent_scales_raw)                  # (M, 3)
        stds = scales_act.repeat_interleave(samples, dim=0)        # (M*S, 3)
        rots = quaternion_to_rotation_matrix(parent_rot_raw)       # (M, 3, 3)
        rots = rots.repeat_interleave(samples, dim=0)
        offsets = torch.normal(
            0.0, 1.0, (total, 3), device=self.device, dtype=parent_pos.dtype
        ) * stds
        offsets = torch.bmm(rots, offsets.unsqueeze(-1)).squeeze(-1)

        child_pos = parent_pos.repeat_interleave(samples, dim=0) + offsets
        child_scales_raw = torch.log(
            scales_act.repeat_interleave(samples, dim=0) / 1.6
        )
        child_rot_raw = parent_rot_raw.repeat_interleave(samples, dim=0)
        child_opac_raw = parent_opac_raw.repeat_interleave(samples, dim=0)
        child_sh = parent_sh.repeat_interleave(samples, dim=0)
        return child_pos, child_scales_raw, child_rot_raw, child_opac_raw, child_sh

    def _thin_mask(self, mask: torch.Tensor, keep_count: int) -> torch.Tensor:
        """Randomly thins a boolean mask down to at most keep_count entries."""
        n = int(mask.sum().item())
        if keep_count >= n:
            return mask
        thinned = torch.zeros_like(mask)
        if keep_count > 0:
            idx = torch.nonzero(mask).squeeze(1)
            perm = torch.randperm(n, device=mask.device)[:keep_count]
            thinned[idx[perm]] = True
        return thinned

    def _apply_gaussians(
        self, new_tensors: Dict[str, torch.Tensor], row_source: torch.Tensor
    ) -> None:
        """Swaps in rebuilt Gaussian tensors and migrates all dependent state.

        Args:
            new_tensors: New dense tensors for each Gaussian parameter.
            row_source: LongTensor where row_source[i] is the old row that new
                row i descends from. Used to gather optimizer moments and the
                is_dynamic buffer so everything stays aligned.
        """
        old_n = self.model._positions.shape[0]
        for name in _PARAM_NAMES:
            old_param = getattr(self.model, name)
            new_param = nn.Parameter(new_tensors[name].contiguous())
            setattr(self.model, name, new_param)

            # Update optimizer parameter reference and gather moments by row
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
                                elif (
                                    torch.is_tensor(v)
                                    and v.ndim > 0
                                    and v.shape[0] == old_n
                                ):
                                    new_state[k] = v[row_source].contiguous()
                                else:
                                    new_state[k] = v
                            self.optimizer.state[new_param] = new_state
                        break

        if hasattr(self.model, "is_dynamic"):
            self.model.register_buffer(
                "is_dynamic", self.model.is_dynamic[row_source].contiguous()
            )
