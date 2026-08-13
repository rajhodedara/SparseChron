import json
import torch
import pathlib
from typing import Dict, List, Union
from sparsechron.evaluation.metrics import Evaluator

def compute_metrics(
    pred_images: List[torch.Tensor], 
    gt_images: List[torch.Tensor]
) -> Dict[str, float]:
    """Computes mean PSNR, SSIM, and LPIPS over pairs of predicted and ground truth images.
    
    Args:
        pred_images (List[torch.Tensor]): List of predicted image tensors (C, H, W) in [0, 1].
        gt_images (List[torch.Tensor]): List of ground truth image tensors (C, H, W) in [0, 1].
        
    Returns:
        Dict[str, float]: Dictionary containing mean metrics ('psnr', 'ssim', 'lpips').
    """
    if len(pred_images) != len(gt_images):
        raise ValueError("Number of predicted images and ground truth images must be equal.")
        
    evaluator = Evaluator()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_images = len(pred_images)
    
    if num_images == 0:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for pred, gt in zip(pred_images, gt_images):
        pred_b = pred.unsqueeze(0).to(device)
        gt_b = gt.unsqueeze(0).to(device)
        
        total_psnr += evaluator.compute_psnr(pred_b, gt_b)
        total_ssim += evaluator.compute_ssim(pred_b, gt_b)
        total_lpips += evaluator.compute_lpips(pred_b, gt_b)
        
        del pred_b, gt_b
        
    mean_psnr = total_psnr / num_images
    mean_ssim = total_ssim / num_images
    mean_lpips = total_lpips / num_images
    
    return {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "lpips": mean_lpips
    }

def save_metrics(metrics: Dict[str, float], output_path: Union[str, pathlib.Path]) -> None:
    """Saves metrics to a JSON file.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics.
        output_path (Union[str, pathlib.Path]): Path to save the metrics.json file.
    """
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=4)
