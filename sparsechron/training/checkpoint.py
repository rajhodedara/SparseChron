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
        
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(
    path: Union[Path, str],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    deformation_mlp: Optional[nn.Module] = None,
    classifier: Optional[Any] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Loads a training checkpoint.
    
    Args:
        path: The path to load the checkpoint from.
        model: The model to load weights into.
        optimizer: The optimizer to load state into (optional).
        deformation_mlp: The deformation MLP to load weights into (optional).
        classifier: The StaticDynamicClassifier to load state into (optional).
        
    Returns:
        A tuple containing the iteration and the extra state dictionary.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if deformation_mlp and "deformation_mlp_state_dict" in checkpoint:
        deformation_mlp.load_state_dict(checkpoint["deformation_mlp_state_dict"])
    if classifier and "classifier_cum_deformation" in checkpoint:
        classifier.cum_deformation = checkpoint["classifier_cum_deformation"].to(classifier.cum_deformation.device)
        
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
