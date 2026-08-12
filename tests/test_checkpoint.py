import torch
import torch.nn as nn
from sparsechron.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

from pathlib import Path

def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Modify model weights to check if they are restored
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        
    checkpoint_path = tmp_path / "checkpoint_100.ckpt"
    
    extra_state = {"epoch": 5}
    save_checkpoint(checkpoint_path, iteration=100, model=model, optimizer=optimizer, extra_state=extra_state)
    
    assert checkpoint_path.exists()
    
    new_model = DummyModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
    
    loaded_iter, loaded_extra = load_checkpoint(checkpoint_path, new_model, new_optimizer)
    
    assert loaded_iter == 100
    assert loaded_extra == extra_state
    
    assert torch.allclose(new_model.linear.weight, torch.ones_like(new_model.linear.weight))

def test_find_latest_checkpoint(tmp_path: Path) -> None:
    # Create dummy checkpoints
    (tmp_path / "checkpoint_100.ckpt").touch()
    (tmp_path / "checkpoint_500.ckpt").touch()
    (tmp_path / "checkpoint_200.ckpt").touch()
    
    # Should find checkpoint_500.ckpt based on iteration number in filename
    latest = find_latest_checkpoint(tmp_path)
    
    assert latest is not None
    assert latest.name == "checkpoint_500.ckpt"
