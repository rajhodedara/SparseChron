"""Interactive viewer application."""
import threading
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import viser
import viser.transforms as vtf

from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.deformation import DeformationMLP
from sparsechron.models.classifier import StaticDynamicClassifier
from sparsechron.models.renderer import GaussianRenderer
from sparsechron.utils.camera import Camera


class ViewerApp:
    """Viewer application for SparseChron using viser."""

    def __init__(
        self,
        model: GaussianModel,
        deformation_mlp: DeformationMLP,
        classifier: StaticDynamicClassifier,
        port: int = 8080,
    ) -> None:
        """Initializes the ViewerApp.

        Args:
            model (GaussianModel): The Gaussian model to render.
            deformation_mlp (DeformationMLP): The deformation MLP.
            classifier (StaticDynamicClassifier): The static/dynamic classifier.
            port (int): The port to run the viewer on.
        """
        self.model = model
        self.deformation_mlp = deformation_mlp
        self.classifier = classifier
        self.device = model.positions.device
        self.renderer = GaussianRenderer()

        self.server = viser.ViserServer(port=port)

        # GUI elements
        with self.server.add_gui_folder("Controls"):
            self.timestep_slider = self.server.add_gui_slider(
                "Timestep", min=0.0, max=1.0, step=0.01, initial_value=0.0
            )
            self.play_button = self.server.add_gui_button("Play / Pause")
            self.bg_checkbox = self.server.add_gui_checkbox(
                "Show Static Background", initial_value=True
            )

        self.stats_text = self.server.add_gui_markdown("FPS: 0.0\n\nGaussians: 0")

        self.playing = False
        self.need_update = True

        @self.play_button.on_click
        def _(_) -> None:
            self.playing = not self.playing

        @self.timestep_slider.on_update
        def _(_) -> None:
            self.need_update = True

        @self.bg_checkbox.on_update
        def _(_) -> None:
            self.need_update = True

        @self.server.on_client_connect
        def on_client_connect(client: viser.ClientHandle) -> None:
            @client.camera.on_update
            def on_camera_update(_) -> None:
                self.need_update = True

        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

    def _render_loop(self) -> None:
        """Main rendering loop running in a separate thread."""
        last_time = time.time()
        frames = 0
        fps_update_time = time.time()

        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if self.playing:
                new_timestep = self.timestep_slider.value + dt * 0.5  # loop in 2 secs
                if new_timestep > 1.0:
                    new_timestep = 0.0
                self.timestep_slider.value = new_timestep
                self.need_update = True

            if self.need_update:
                clients = self.server.get_clients()
                for client_id, client in clients.items():
                    cam = client.camera

                    # Viser camera (c2w, OpenGL convention)
                    c2w_opengl = np.eye(4)
                    c2w_opengl[:3, :3] = vtf.SO3(cam.wxyz).as_matrix()
                    c2w_opengl[:3, 3] = cam.position

                    # Convert to OpenCV convention for gsplat
                    c2w_opencv = c2w_opengl.copy()
                    c2w_opencv[:3, 1:3] *= -1
                    w2c = np.linalg.inv(c2w_opencv)

                    height = 512
                    width = int(height * cam.aspect)
                    fx = 0.5 * height / np.tan(cam.fov / 2.0)
                    fy = fx
                    cx = width / 2.0
                    cy = height / 2.0

                    camera = Camera(
                        fx=float(fx),
                        fy=float(fy),
                        cx=float(cx),
                        cy=float(cy),
                        width=width,
                        height=height,
                        R=torch.tensor(
                            w2c[:3, :3], dtype=torch.float32, device=self.device
                        ),
                        T=torch.tensor(
                            w2c[:3, 3], dtype=torch.float32, device=self.device
                        ),
                    )

                    with torch.no_grad():
                        deformed_params = self.model.get_deformed(
                            self.deformation_mlp, self.timestep_slider.value
                        )

                        if not self.bg_checkbox.value:
                            static_mask = ~self.model.is_dynamic
                            deformed_params["opacities"][static_mask] = 0.0

                        out = self.renderer.render(
                            self.model, camera, deformed_params
                        )
                        rgb = out["rgb"]

                    rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    client.set_background_image(rgb_np, format="jpeg")

                    # Update stats
                    if not self.bg_checkbox.value:
                        active_gaussians = int(self.model.is_dynamic.sum().item())
                    else:
                        active_gaussians = int(self.model.positions.shape[0])

                    frames += 1
                    if current_time - fps_update_time > 1.0:
                        fps = frames / (current_time - fps_update_time)
                        self.stats_text.content = (
                            f"**FPS:** {fps:.1f}\n\n**Gaussians:** {active_gaussians}"
                        )
                        frames = 0
                        fps_update_time = current_time

                self.need_update = False

            time.sleep(1.0 / 60.0)
