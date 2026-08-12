import torch
import torch.nn.functional as F
from math import exp

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian tensor.
    
    Args:
        window_size: The size of the Gaussian window.
        sigma: The standard deviation of the Gaussian.
        
    Returns:
        A 1D tensor representing the Gaussian.
    """
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian window for SSIM computation.
    
    Args:
        window_size: The size of the window.
        channel: The number of channels.
        
    Returns:
        A 4D tensor representing the window.
    """
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True
) -> torch.Tensor:
    """Computes the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: The first image tensor (H, W, C) or (C, H, W) or (B, C, H, W).
        img2: The second image tensor (H, W, C) or (C, H, W) or (B, C, H, W).
        window_size: The size of the Gaussian window.
        size_average: If True, returns the mean SSIM.
        
    Returns:
        The SSIM value.
    """
    if img1.dim() == 3:
        if img1.shape[2] in [1, 3]:  # H, W, C
            img1 = img1.permute(2, 0, 1).unsqueeze(0)
            img2 = img2.permute(2, 0, 1).unsqueeze(0)
        else:  # C, H, W
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    window = window.to(device=img1.device, dtype=img1.dtype)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Computes the L1 loss between prediction and ground truth.
    
    Args:
        pred: The predicted image tensor.
        gt: The ground truth image tensor.
        
    Returns:
        The L1 loss.
    """
    return torch.abs(pred - gt).mean()

def photometric_loss(
    pred: torch.Tensor, gt: torch.Tensor, lambda_ssim: float
) -> torch.Tensor:
    """Computes the combined L1 and SSIM loss.
    
    Args:
        pred: The predicted image tensor.
        gt: The ground truth image tensor.
        lambda_ssim: The weight for the SSIM loss.
        
    Returns:
        The combined loss.
    """
    assert isinstance(pred, torch.Tensor), "pred must be a torch.Tensor"
    assert isinstance(gt, torch.Tensor), "gt must be a torch.Tensor"
    if not (0.0 <= lambda_ssim <= 1.0):
        raise ValueError("lambda_ssim must be between 0.0 and 1.0")
        
    return (1.0 - lambda_ssim) * l1_loss(pred, gt) + lambda_ssim * (1.0 - ssim(pred, gt))
