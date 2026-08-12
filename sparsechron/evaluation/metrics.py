import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

class Evaluator:
    """Evaluates image quality using PSNR, SSIM, and LPIPS metrics.
    
    Attributes:
        device (torch.device): The device to run the metrics on.
        psnr (PeakSignalNoiseRatio): The PSNR metric object.
        ssim (StructuralSimilarityIndexMeasure): The SSIM metric object.
        lpips_fn (lpips.LPIPS): The LPIPS metric object.
    """

    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        """Initializes the Evaluator.
        
        Args:
            device (torch.device, optional): Device to compute metrics on. Defaults to cuda if available.
        """
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device)

    def compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Computes the Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            pred (torch.Tensor): Predicted image tensor of shape (B, 3, H, W) normalized to [0, 1].
            gt (torch.Tensor): Ground truth image tensor of shape (B, 3, H, W) normalized to [0, 1].
            
        Returns:
            float: The PSNR value.
        """
        return self.psnr(pred, gt).item()

    def compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Computes the Structural Similarity Index Measure (SSIM).
        
        Args:
            pred (torch.Tensor): Predicted image tensor of shape (B, 3, H, W) normalized to [0, 1].
            gt (torch.Tensor): Ground truth image tensor of shape (B, 3, H, W) normalized to [0, 1].
            
        Returns:
            float: The SSIM value.
        """
        return self.ssim(pred, gt).item()

    def compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Computes the Learned Perceptual Image Patch Similarity (LPIPS).
        
        Args:
            pred (torch.Tensor): Predicted image tensor of shape (B, 3, H, W) normalized to [0, 1].
            gt (torch.Tensor): Ground truth image tensor of shape (B, 3, H, W) normalized to [0, 1].
            
        Returns:
            float: The LPIPS value.
        """
        pred_scaled = pred * 2.0 - 1.0
        gt_scaled = gt * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = self.lpips_fn(pred_scaled, gt_scaled)
        return lpips_val.mean().item()
