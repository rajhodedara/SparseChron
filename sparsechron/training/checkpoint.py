import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

def save_checkpoint(
    path: Union[Path, str],
    iteration: int,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    deformation_mlp: Optional[nn.Module] = None,
    classifier: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> None:
    """Saves a training checkpoint atomically.

    Args:
        path: The path to save the checkpoint to.
        iteration: The current training iteration.
        model: The model to save.
        optimizer: The optimizer to save (optional).
        extra_state: Additional state to save (optional).
        deformation_mlp: The deformation MLP to save (optional).
        classifier: The StaticDynamicClassifier to save (optional).
        scaler: The GradScaler to save (optional, mixed precision resume).
    """
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")

    state = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "extra_state": extra_state or {},
    }

    if deformation_mlp is not None:
        state["deformation_mlp_state_dict"] = deformation_mlp.state_dict()
    if classifier is not None:
        state["classifier_cum_deformation"] = classifier.cum_deformation
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(
    path: Union[Path, str],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    deformation_mlp: Optional[nn.Module] = None,
    classifier: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Loads a training checkpoint.

    Args:
        path: The path to load the checkpoint from.
        model: The model to load weights into.
        optimizer: The optimizer to load state into (optional).
        deformation_mlp: The deformation MLP to load weights into (optional).
        classifier: The StaticDynamicClassifier to load state into (optional).
        scaler: The GradScaler to load state into (optional).

    Returns:
        A tuple containing the iteration and the extra state dictionary.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")

    # Handle size mismatch by resizing GaussianModel parameters and buffers if needed
    model_state = checkpoint["model_state_dict"]
    for name, param in list(model.named_parameters()):
        if name in model_state:
            chk_shape = model_state[name].shape
            if param.shape != chk_shape:
                new_param = nn.Parameter(torch.zeros(chk_shape, dtype=param.dtype, device=param.device))
                setattr(model, name, new_param)

                # Update references in optimizer
                if optimizer:
                    for group in optimizer.param_groups:
                        for i, p in enumerate(group['params']):
                            if p is param:
                                group['params'][i] = new_param
                                # Migrate state
                                if p in optimizer.state:
                                    optimizer.state[new_param] = optimizer.state.pop(p)
                                break

    for name, buf in list(model.named_buffers()):
        if name in model_state:
            chk_shape = model_state[name].shape
            if buf.shape != chk_shape:
                new_buf = torch.zeros(chk_shape, dtype=buf.dtype, device=buf.device)
                setattr(model, name, new_buf)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if deformation_mlp and "deformation_mlp_state_dict" in checkpoint:
        deformation_mlp.load_state_dict(checkpoint["deformation_mlp_state_dict"])
    if classifier and "classifier_cum_deformation" in checkpoint:
        classifier.cum_deformation = checkpoint["classifier_cum_deformation"].to(classifier.cum_deformation.device)
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint.get("iteration", 0), checkpoint.get("extra_state", {})

def find_latest_checkpoint(directory: Union[Path, str]) -> Optional[Path]:
    """Finds the latest checkpoint in a directory based on iteration number.

    Args:
        directory: The directory to search.

    Returns:
        The path to the latest checkpoint, or None if no checkpoints are found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    checkpoints = list(directory.glob("*.ckpt"))
    if not checkpoints:
        return None

    def extract_iter(p: Path) -> Tuple[int, float]:
        parts = p.stem.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return (1, int(part))
        return (0, p.stat().st_mtime)

    return max(checkpoints, key=extract_iter)

def prune_checkpoints(directory: Union[Path, str], keep_last_k: int = 3) -> None:
    """Deletes older checkpoints, keeping only the newest K by iteration number.

    Prevents /kaggle/working from filling up during long Kaggle sessions.

    Args:
        directory: The checkpoint directory.
        keep_last_k: Number of newest checkpoints to keep (<= 0 disables pruning).
    """
    if keep_last_k <= 0:
        return
    directory = Path(directory)
    if not directory.is_dir():
        return

    entries = []
    for p in directory.glob("checkpoint_*.ckpt"):
        for part in reversed(p.stem.split("_")):
            if part.isdigit():
                entries.append((int(part), p))
                break

    entries.sort(key=lambda t: t[0])
    for _, p in entries[:-keep_last_k]:
        try:
            p.unlink()
        except OSError:
            # Best effort; never fail training because an old file is locked.
            pass
