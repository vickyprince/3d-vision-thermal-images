# utils/metrics.py
import torch
import numpy as np

def compute_depth_metrics(pred, gt, mask=None):
    """
    Compute depth estimation metrics
    
    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
        mask: Optional mask for valid pixels
        
    Returns:
        Dictionary of metrics
    """
    if mask is None:
        # Create mask for valid depth values
        mask = (gt > 0)
    else:
        mask = mask & (gt > 0)
    
    # Apply mask
    pred = pred[mask]
    gt = gt[mask]
    
    # If no valid pixels, return zeros
    if mask.sum() == 0:
        return {
            'rmse': 0.0,
            'abs_rel': 0.0,
            'acc_1.25': 0.0,
            'acc_1.25^2': 0.0,
            'acc_1.25^3': 0.0
        }
    
    # Compute metrics
    thresh = torch.maximum(pred / gt, gt / pred)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    
    # Accuracy metrics (δ < 1.25, 1.25², 1.25³)
    acc_1 = (thresh < 1.25).float().mean()
    acc_2 = (thresh < 1.25 ** 2).float().mean()
    acc_3 = (thresh < 1.25 ** 3).float().mean()
    
    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'acc_1.25': acc_1.item(),
        'acc_1.25^2': acc_2.item(),
        'acc_1.25^3': acc_3.item()
    }