# scripts/evaluate.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.dust3r import DUSt3R
from data.freiburg_dataset import FreiburgThermalDataset
from utils.metrics import compute_depth_metrics
from utils.visualization import visualize_depth_map, visualize_pointcloud

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DUSt3R on thermal data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--vis_samples', type=int, default=10, help='Number of samples to visualize')
    return parser.parse_args()

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config['device'])
    
    # Create dataset and dataloader
    test_dataset = FreiburgThermalDataset(
        root_dir=config['data']['test_path'],
        annotations_path=config['data']['annotations_path'],
        transform=get_transforms(config, 'test')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model = DUSt3R(
        backbone=config['model']['backbone'],
        pretrained=False  # We'll load from checkpoint
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    model.eval()
    all_metrics = {
        'rmse': [],
        'abs_rel': [],
        'acc_1.25': [],
        'acc_1.25^2': [],
        'acc_1.25^3': []
    }
    
    vis_indices = np.random.choice(len(test_dataset), args.vis_samples, replace=False)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            # Move data to device
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            gt_depth = batch['gt_depth'].to(device)
            
            # Forward pass
            outputs = model(img1, img2)
            pred_pointmap1 = outputs['pointmap1']
            
            # Extract depth (Z component of pointmap)
            pred_depth = pred_pointmap1[:, 2, :, :]
            
            # Compute metrics
            metrics = compute_depth_metrics(pred_depth, gt_depth)
            
            # Update metrics
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            # Visualize some samples
            for b in range(img1.shape[0]):
                idx = i * config['evaluation']['batch_size'] + b
                if idx >= len(test_dataset):
                    break
                if idx in vis_indices:
                    # Get original thermal image (for visualization)
                    thermal_img = batch['thermal_orig'][b].cpu().numpy()
                    
                    # Get predicted depth and point cloud
                    depth_pred = pred_depth[b].cpu().numpy()
                    pointmap_pred = pred_pointmap1[b].cpu().numpy()
                    
                    # Get ground truth depth and point cloud
                    depth_gt = gt_depth[b].cpu().numpy()
                    
                    # Create visualization folder
                    sample_dir = Path(args.output_dir) / f"sample_{idx}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    # Save thermal image
                    plt.imsave(str(sample_dir / "thermal.png"), thermal_img, cmap='inferno')
                    
                    # Visualize and save depth maps
                    visualize_depth_map(depth_pred, str(sample_dir / "depth_pred.png"))
                    visualize_depth_map(depth_gt, str(sample_dir / "depth_gt.png"))
                    
                    # Visualize and save point clouds
                    intrinsics = batch['intrinsics'][b].numpy()
                    visualize_pointcloud(pointmap_pred, str(sample_dir / "pointcloud_pred.ply"))
    
    # Compute average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    # Print metrics
    print("Evaluation Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

def get_transforms(config, split):
    # Same as in training script
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

if __name__ == "__main__":
    args = parse_args()
    main(args)