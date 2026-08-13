import pathlib
import tyro
import torch

from sparsechron.models.gaussians import GaussianModel
from sparsechron.models.deformation import DeformationMLP
from sparsechron.data.dataset import SceneDataset
from sparsechron.evaluation.render_novel_views import render_novel_views
from sparsechron.evaluation.benchmark import compute_metrics, save_metrics

def main(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str = "evaluation_output",
    is_4d: bool = True
) -> None:
    """Evaluates a trained SparseChron model on a dataset.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        dataset_path (str): Path to the dataset configuration or directory.
        output_dir (str, optional): Directory to save outputs. Defaults to "evaluation_output".
        is_4d (bool, optional): Whether the model is 4D (includes deformation). Defaults to True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create dummy initial values based on checkpoint size
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    num_gaussians = state_dict['_positions'].shape[0]
    sh_degree = state_dict['_sh_coeffs'].shape[1]
    
    dummy_values = {
        "positions": torch.zeros((num_gaussians, 3), device=device),
        "scales": torch.zeros((num_gaussians, 3), device=device),
        "rotations": torch.zeros((num_gaussians, 4), device=device),
        "opacities": torch.zeros((num_gaussians, 1), device=device),
        "sh_coeffs": torch.zeros((num_gaussians, sh_degree, 3), device=device),
    }
    
    model = GaussianModel(dummy_values)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    deformation_mlp = None
    if is_4d:
        deformation_mlp = DeformationMLP()
        if 'deformation_state_dict' in checkpoint:
            deformation_mlp.load_state_dict(checkpoint['deformation_state_dict'])
        elif 'deformation_mlp_state_dict' in checkpoint:
            deformation_mlp.load_state_dict(checkpoint['deformation_mlp_state_dict'])
        deformation_mlp.to(device)
        deformation_mlp.eval()
        
    print(f"Loading dataset from {dataset_path}...")
    # Our SceneDataset loads all images from the folder, regardless of split.
    dataset = SceneDataset(dataset_path)
    
    from sparsechron.models.renderer import GaussianRenderer
    renderer = GaussianRenderer()
    
    print("Rendering novel views...")
    render_out_dir = pathlib.Path(output_dir) / "renders"
    rendered_results = render_novel_views(
        model=model,
        renderer=renderer,
        dataset=dataset,
        deformation_mlp=deformation_mlp,
        output_dir=str(render_out_dir)
    )
    
    # Extract predicted and ground truth images
    pred_images = []
    gt_images = []
    
    for i, batch in enumerate(dataset):
        if isinstance(batch, tuple) or isinstance(batch, list):
            gt_img = batch[2] if len(batch) > 2 else None
        elif hasattr(batch, 'image'):
            gt_img = batch.image
        else:
            gt_img = None
            
        if gt_img is not None:
            pred_images.append(rendered_results[i]["image"].cpu())
            gt_images.append(gt_img.cpu())
            
    if len(pred_images) > 0 and len(pred_images) == len(gt_images):
        print("Computing metrics...")
        metrics = compute_metrics(pred_images, gt_images)
        
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")
            
        metrics_path = pathlib.Path(output_dir) / "metrics.json"
        save_metrics(metrics, metrics_path)
        print(f"Metrics saved to {metrics_path}")
    else:
        print("No ground truth images found or mismatch in counts. Skipping metrics computation.")

if __name__ == "__main__":
    tyro.cli(main)
