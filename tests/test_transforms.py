import pytest
import torch
from sparsechron.utils.transforms import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def test_normalize_quaternion():
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    q_norm = normalize_quaternion(q)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
    )
    assert torch.allclose(q_norm, expected)

    with pytest.raises(ValueError):
        normalize_quaternion(torch.randn(10, 3))


def test_quaternion_multiply():
    # q1 = 1 + 0i + 0j + 0k
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # q2 = 0 + 1i + 0j + 0k
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    out = quaternion_multiply(q1, q2)
    assert torch.allclose(out, q2)

    # i * j = k -> (0, 1, 0, 0) * (0, 0, 1, 0) = (0, 0, 0, 1)
    qi = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    qj = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    qk = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    out2 = quaternion_multiply(qi, qj)
    assert torch.allclose(out2, qk)

    with pytest.raises(ValueError):
        quaternion_multiply(torch.randn(10, 3), torch.randn(10, 4))


def test_quaternion_to_rotation_matrix_and_back():
    torch.manual_seed(42)
    
    # Generate random quaternions and normalize them
    q = torch.randn(10, 4)
    q = normalize_quaternion(q)

    # q -> R
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (10, 3, 3)

    # R -> q
    q_out = rotation_matrix_to_quaternion(R)
    assert q_out.shape == (10, 4)

    # q and -q represent the same rotation, so test if q_out == q or q_out == -q
    match_pos = torch.allclose(q_out, q, atol=1e-5)
    match_neg = torch.allclose(q_out, -q, atol=1e-5)
    
    # To check properly for batched tensors:
    # Check if either q_out is close to q or q_out is close to -q for each element
    diff_pos = torch.norm(q_out - q, dim=-1)
    diff_neg = torch.norm(q_out + q, dim=-1)
    min_diff = torch.minimum(diff_pos, diff_neg)
    
    assert torch.all(min_diff < 1e-4)

    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(torch.randn(10, 3))
    
    with pytest.raises(ValueError):
        rotation_matrix_to_quaternion(torch.randn(10, 3, 2))
