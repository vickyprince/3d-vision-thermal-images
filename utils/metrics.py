import torch
import numpy as np

def compute_depth_metrics(pred, gt, mask=None):
    """
    Compute depth estimation metrics between predicted and ground truth depth maps.

    Metrics computed:
      - RMSE (Root Mean Squared Error)
      - Abs Rel (Absolute Relative Error)
      - Accuracy with thresholds δ < 1.25, δ < 1.25^2, δ < 1.25^3

    Args:
        pred (torch.Tensor): Predicted depth map.
        gt (torch.Tensor): Ground truth depth map.
        mask (torch.Tensor, optional): Boolean mask for valid pixels. If not provided, valid pixels are those where gt > 0.

    Returns:
        dict: A dictionary containing:
            'rmse': float,
            'abs_rel': float,
            'acc_1.25': float,
            'acc_1.25^2': float,
            'acc_1.25^3': float
    """
    if mask is None:
        mask = (gt > 0)
    else:
        mask = mask & (gt > 0)

    pred = pred[mask]
    gt = gt[mask]

    if mask.sum() == 0:
        return {
            'rmse': 0.0,
            'abs_rel': 0.0,
            'acc_1.25': 0.0,
            'acc_1.25^2': 0.0,
            'acc_1.25^3': 0.0
        }

    thresh = torch.maximum(pred / gt, gt / pred)
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
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