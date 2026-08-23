import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sparsechron.utils.config import TrainConfig
from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.renderer import GaussianRenderer

class SimpleCamera:
    def __init__(self, R, T, width, height, fx, fy, cx, cy):
        self.R = torch.tensor(R, dtype=torch.float32, device="cuda")
        self.T = torch.tensor(T, dtype=torch.float32, device="cuda")
        self.width = int(width)
        self.height = int(height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def load_models(config, checkpoint_path):
    print("Loading models...")
    device = torch.device("cuda")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    num_gaussians = state_dict["_positions"].shape[0]
    sh_degree = state_dict["_sh_coeffs"].shape[1]
    
    dummy_values = {
        "positions": torch.zeros((num_gaussians, 3), device=device),
        "scales": torch.zeros((num_gaussians, 3), device=device),
        "rotations": torch.zeros((num_gaussians, 4), device=device),
        "opacities": torch.zeros((num_gaussians, 1), device=device),
        "sh_coeffs": torch.zeros((num_gaussians, sh_degree, 3), device=device),
    }
    
    model = GaussianModel(dummy_values)
    model.load_state_dict(state_dict, strict=False)
    if "is_dynamic" in state_dict:
        model.is_dynamic = state_dict["is_dynamic"].to(device)
    model.to(device)
    model.eval()
    
    # Setup deformation MLP
    deform = DeformationMLP()
    if "deformation_mlp_state_dict" in ckpt:
        deform.load_state_dict(ckpt["deformation_mlp_state_dict"])
    elif "deformation_state_dict" in ckpt:
        deform.load_state_dict(ckpt["deformation_state_dict"])
    deform.to(device)
    deform.eval()
    
    renderer = GaussianRenderer()
    
    print("Models loaded successfully!")
    return model, deform, renderer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config = TrainConfig()
    
    # Initialize CUDA if available
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Viser viewer requires GPU.")
        return

    model, deform, renderer = load_models(config, args.checkpoint)
    
    server = viser.ViserServer(port=args.port)
    print(f"\nViser server running on port {args.port}")
    print(f"Run `npx localtunnel --port {args.port}` in another cell to get a public link!\n")
    
    # Add UI elements
    # Add UI elements
    with server.gui.add_folder("4D Controls"):
        time_slider = server.gui.add_slider("Time", min=0.0, max=1.0, step=0.01, initial_value=0.0)
        res_slider = server.gui.add_slider("Resolution", min=256, max=1024, step=128, initial_value=512)
        
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print("Client connected!")
        
        def render_and_send():
            # Get current camera pose from Viser
            cam = client.camera
            
            # Viser cameras already follow the OpenCV convention (+Z forward, +Y down, +X right).
            R_c2w = vtf.SO3(cam.wxyz).as_matrix()
            T_c2w = np.array(cam.position)
            
            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = T_c2w
            
            # We can directly invert c2w to get the w2c matrix expected by gsplat.
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            
            # Resolution and intrinsics
            width = res_slider.value
            height = int(width / cam.aspect)
            
            # Calculate focal length from vertical FOV
            fy = (height / 2.0) / np.tan(cam.fov / 2.0)
            fx = fy # Assume square pixels
            
            cx = width / 2.0
            cy = height / 2.0
            
            camera = SimpleCamera(R, T, width, height, fx, fy, cx, cy)
            
            timestep = time_slider.value
            
            with torch.no_grad():
                deformed_params = model.get_deformed(deform, timestep)
                
                # Render
                render_dict = renderer.render(model, camera, deformed_params=deformed_params)
                
                # Convert to uint8 numpy array
                img_tensor = render_dict["rgb"].clamp(0, 1) # [H, W, 3]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                # Send to Viser as background
                client.scene.set_background_image(img_np, format="jpeg")

        # Attach callbacks
        @client.camera.on_update
        def _(_):
            render_and_send()
            
        @time_slider.on_update
        def _(_):
            render_and_send()
            
        @res_slider.on_update
        def _(_):
            render_and_send()
            
        # Force initial render so the screen isn't black
        render_and_send()

    # Keep main thread alive
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
