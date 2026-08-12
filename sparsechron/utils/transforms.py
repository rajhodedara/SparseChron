"""Coordinate transforms and mathematical utilities."""

import torch


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalizes quaternions to unit length.
    
    Args:
        q (torch.Tensor): Tensor of shape (N, 4) representing quaternions.
        
    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing normalized quaternions.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"q must have shape (..., 4), got {q.shape}")
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiplies two sets of quaternions.
    
    Uses the wxyz convention where the real part is at index 0.
    
    Args:
        q1 (torch.Tensor): Tensor of shape (N, 4).
        q2 (torch.Tensor): Tensor of shape (N, 4).
        
    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing the product q1 * q2.
    """
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(f"q1 and q2 must have shape (..., 4), got {q1.shape} and {q2.shape}")
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Converts quaternions to rotation matrices.
    
    Uses the wxyz convention where the real part is at index 0.
    
    Args:
        q (torch.Tensor): Tensor of shape (N, 4) representing quaternions.
        
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) representing rotation matrices.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"q must have shape (..., 4), got {q.shape}")
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)
    
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrices to quaternions.
    
    Uses the wxyz convention where the real part is at index 0.
    Implementation avoids taking the square root of negative numbers due to numerical
    instability by using four different branches based on the trace and diagonal elements.
    
    Args:
        R (torch.Tensor): Tensor of shape (N, 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing quaternions in wxyz format.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3), got {R.shape}")
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    
    trace = m00 + m11 + m22
    
    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0.0))
        
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    
    cond1 = trace > 0
    cond2 = (m00 > m11) & (m00 > m22)
    cond3 = m11 > m22
    
    # branch 1
    S1 = torch.clamp(safe_sqrt(1.0 + trace) * 2.0, min=1e-8)
    q_w1 = 0.25 * S1
    q_x1 = (m21 - m12) / S1
    q_y1 = (m02 - m20) / S1
    q_z1 = (m10 - m01) / S1
    
    # branch 2
    S2 = torch.clamp(safe_sqrt(1.0 + m00 - m11 - m22) * 2.0, min=1e-8)
    q_w2 = (m21 - m12) / S2
    q_x2 = 0.25 * S2
    q_y2 = (m01 + m10) / S2
    q_z2 = (m02 + m20) / S2
    
    # branch 3
    S3 = torch.clamp(safe_sqrt(1.0 + m11 - m00 - m22) * 2.0, min=1e-8)
    q_w3 = (m02 - m20) / S3
    q_x3 = (m01 + m10) / S3
    q_y3 = 0.25 * S3
    q_z3 = (m12 + m21) / S3
    
    # branch 4
    S4 = torch.clamp(safe_sqrt(1.0 + m22 - m00 - m11) * 2.0, min=1e-8)
    q_w4 = (m10 - m01) / S4
    q_x4 = (m02 + m20) / S4
    q_y4 = (m12 + m21) / S4
    q_z4 = 0.25 * S4
    
    q[cond1] = torch.stack([q_w1, q_x1, q_y1, q_z1], dim=-1)[cond1]
    
    mask2 = (~cond1) & cond2
    q[mask2] = torch.stack([q_w2, q_x2, q_y2, q_z2], dim=-1)[mask2]
    
    mask3 = (~cond1) & (~cond2) & cond3
    q[mask3] = torch.stack([q_w3, q_x3, q_y3, q_z3], dim=-1)[mask3]
    
    mask4 = (~cond1) & (~cond2) & (~cond3)
    q[mask4] = torch.stack([q_w4, q_x4, q_y4, q_z4], dim=-1)[mask4]
    
    return normalize_quaternion(q)
