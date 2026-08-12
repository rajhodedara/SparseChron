import torch
import sys
sys.path.append(r"c:\Users\odeda\Desktop\Projects\IVP Project")
from sparsechron.utils.camera import Camera
from sparsechron.utils.transforms import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, normalize_quaternion, quaternion_multiply

def test_transforms():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
    q = normalize_quaternion(q)
    R = quaternion_to_rotation_matrix(q)
    q2 = rotation_matrix_to_quaternion(R)
    print("q:", q)
    print("q2:", q2)
    assert torch.allclose(torch.abs(q), torch.abs(q2), atol=1e-4)
    print("transforms pass!")

def test_camera():
    R = torch.eye(3)
    T = torch.tensor([0.0, 0.0, 1.0])
    cam = Camera(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100, R=R, T=T)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cam_points = cam.world_to_camera(points)
    print("cam_points:", cam_points)
    proj = cam.project(points)
    print("proj:", proj)
    
    depth = torch.ones((2, 2))
    unproj = cam.unproject(depth)
    print("unproj shape:", unproj.shape)
    print("camera pass!")

if __name__ == "__main__":
    test_transforms()
    test_camera()
