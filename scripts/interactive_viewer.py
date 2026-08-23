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
        self.image_width = int(width)
        self.image_height = int(height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def load_models(config, checkpoint_path):
    print("Loading models...")
    model = GaussianModel(config.sh_degree)
    
    # Setup deformation MLP
    deform = DeformationMLP(
        D=8, W=256,
        input_ch=3, input_ch_time=1,
        skips=[4],
        empty_voxel=False,
        is_blender=True # Ensure correct positional encoding
    ).to("cuda")
    
    renderer = GaussianRenderer(config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Load Gaussian params
    model.restore(ckpt["model_state_dict"], config)
    
    # Load deformation MLP weights
    deform.load_state_dict(ckpt["deform_state_dict"])
    
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
    with server.gui.add_folder("4D Controls"):
        time_slider = server.gui.add_slider("Time", min=0.0, max=1.0, step=0.01, initial_value=0.0)
        res_slider = server.gui.add_slider("Resolution", min=256, max=1024, step=128, initial_value=512)
        
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print("Client connected!")
        
        # Turn off grid for better viewing of the background image
        client.scene.world_axes.visible = False
        client.scene.grid.visible = False
        
        def render_and_send():
            # Get current camera pose from Viser
            cam = client.camera
            
            # Viser provides c2w quaternion and position (OpenGL format: Y up, Z backward)
            R_c2w = vtf.SO3(cam.wxyz).as_matrix()
            T_c2w = np.array(cam.position)
            
            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = T_c2w
            
            # Convert to OpenCV (w2c, Y down, Z forward)
            c2w_cv = c2w.copy()
            c2w_cv[:, 1:3] *= -1 # Flip Y and Z
            
            w2c = np.linalg.inv(c2w_cv)
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
            
            # Query Deformation MLP
            with torch.no_grad():
                # Pass all points to MLP (or use static/dynamic mask if needed)
                d_xyz, d_rot, d_scale = deform(
                    model.get_xyz, 
                    torch.full((model.get_xyz.shape[0], 1), timestep, device="cuda")
                )
                
                deformed_params = {
                    "xyz": model.get_xyz + d_xyz,
                    "rotation": model.get_rotation + d_rot,
                    "scaling": model.get_scaling + d_scale
                }
                
                # Render
                render_dict = renderer.render(model, camera, deformed_params=deformed_params)
                
                # Convert to uint8 numpy array
                img_tensor = render_dict["render"].clamp(0, 1) # [3, H, W]
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Send to Viser as background
                client.set_background_image(img_np, format="jpeg", depth=1.0)

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

    # Keep main thread alive
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
