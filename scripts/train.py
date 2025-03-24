#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F

# Disable the symmetrization check to avoid ambiguous tensor comparisons.
from dust3r.utils import misc
misc.is_symmetrized = lambda x, y: False

from models.dust3r import DUSt3R
from data.freiburg_train_dataset import FreiburgTrainThermalDataset
from utils.metrics import compute_depth_metrics
from utils.augmentation import ThermalAugmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Train DUSt3R on thermal data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def compute_thermal_stats(dataset, num_samples=100):
    """
    Compute actual mean and std from thermal dataset for better normalization.
    """
    from torch.utils.data import DataLoader
    import random
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    print("Computing thermal image statistics...")
    for batch in tqdm(loader):
        img = batch['img1']  # [B, 3, H, W]
        mean += img.mean(dim=[0, 2, 3])
    mean /= len(indices)
    
    for batch in tqdm(loader):
        img = batch['img1']
        for c in range(3):
            std[c] += ((img[:, c, :, :] - mean[c])**2).mean()
    std = torch.sqrt(std / len(indices))
    
    return mean.tolist(), std.tolist()

class DepthLoss(nn.Module):
    """
    Combined loss function for depth estimation with scale-invariant components.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.05):
        """
        Reduced gamma from 0.1 to 0.05 to reduce the edge-aware smoothness penalty.
        Increased alpha to 1.0 so that L1 has more weight relative to scale-invariant loss.
        """
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target, img=None):
        # Basic L1 loss
        l1_loss = torch.abs(pred - target).mean()

        # Ensure pred and target are strictly positive
        pred_clamped = torch.clamp(pred, min=1e-6)
        target_clamped = torch.clamp(target, min=1e-6)

        # Scale-invariant loss (from Eigen et al.)
        n = pred.numel()
        log_diff = torch.log(pred_clamped) - torch.log(target_clamped)
        scale_inv_loss = (log_diff**2).sum() / n - (log_diff.sum()**2) / (n**2)

        # Edge-aware smoothness loss
        smoothness_loss = 0
        if img is not None and self.gamma > 0:
            img_grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
            img_grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
            img_grad_x = img_grad_x.mean(dim=1, keepdim=True)
            img_grad_y = img_grad_y.mean(dim=1, keepdim=True)
            
            depth_grad_x = torch.abs(pred_clamped[:, :, :, :-1] - pred_clamped[:, :, :, 1:])
            depth_grad_y = torch.abs(pred_clamped[:, :, :-1, :] - pred_clamped[:, :, 1:, :])
            
            weights_x = torch.exp(-img_grad_x)
            weights_y = torch.exp(-img_grad_y)
            
            smoothness_loss = (depth_grad_x * weights_x).mean() + (depth_grad_y * weights_y).mean()
        
        total_loss = self.alpha * l1_loss + self.beta * scale_inv_loss + self.gamma * smoothness_loss

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'scale_inv_loss': scale_inv_loss.item(),
            'smoothness_loss': smoothness_loss.item() if not isinstance(smoothness_loss, float) else smoothness_loss
        }

class PointMapLoss(nn.Module):
    """
    Loss function for pointmap prediction with consistency constraints.
    """
    def __init__(self, w_l1=1.0, w_consistency=0.5):
        super(PointMapLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_consistency = w_consistency
        # Use the revised DepthLoss with adjusted alpha/gamma
        self.depth_loss = DepthLoss(alpha=1.0, beta=0.5, gamma=0.05)
        
    def forward(self, pred_pm1, pred_pm2, gt_pm1, gt_pm2, img1=None, img2=None, pose1=None, pose2=None):
        # L1 for entire pointmap
        l1_pm1 = F.l1_loss(pred_pm1, gt_pm1)
        l1_pm2 = F.l1_loss(pred_pm2, gt_pm2)
        l1_loss = (l1_pm1 + l1_pm2) / 2.0
        
        # Depth channel
        pred_depth1 = pred_pm1[:, 2:3]
        pred_depth2 = pred_pm2[:, 2:3]
        gt_depth1 = gt_pm1[:, 2:3]
        gt_depth2 = gt_pm2[:, 2:3]
        
        depth_loss1, _ = self.depth_loss(pred_depth1, gt_depth1, img1)
        depth_loss2, _ = self.depth_loss(pred_depth2, gt_depth2, img2)
        depth_loss = (depth_loss1 + depth_loss2) / 2.0
        
        # Consistency loss
        consistency_loss = 0
        if pose1 is not None and pose2 is not None and self.w_consistency > 0:
            B, _, H, W = pred_pm1.shape
            # Flatten pointmap1 for each sample
            points1 = pred_pm1.permute(0, 2, 3, 1).reshape(B, -1, 3)
            ones = torch.ones(B, H * W, 1, device=points1.device)
            points1_homogeneous = torch.cat([points1, ones], dim=2)
            
            rel_pose = torch.bmm(pose2, torch.inverse(pose1))
            points1_in_cam2 = torch.bmm(points1_homogeneous, rel_pose.transpose(1, 2))[:, :, :3]
            points1_in_cam2 = points1_in_cam2.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            
            mask = (pred_pm1[:, 2:3] > 0) & (points1_in_cam2[:, 2:3] > 0)
            if mask.sum() > 0:
                B, C, H, W = points1_in_cam2.shape
                points1_flat = points1_in_cam2.permute(0, 2, 3, 1).reshape(-1, 3)
                points2_flat = pred_pm2.permute(0, 2, 3, 1).reshape(-1, 3)
                mask_flat = mask.view(-1)
                valid_points1 = points1_flat[mask_flat]
                valid_points2 = points2_flat[mask_flat]
                
                # Lower max_points to reduce memory usage & possible OOM
                max_points = 2000  # was 5000
                if valid_points1.size(0) > max_points:
                    indices = torch.randperm(valid_points1.size(0), device=valid_points1.device)[:max_points]
                    valid_points1 = valid_points1[indices]
                    valid_points2 = valid_points2[indices]
                
                consistency_loss = F.l1_loss(valid_points1, valid_points2)
            else:
                consistency_loss = torch.tensor(0.0, device=pred_pm1.device)
        
        total_loss = self.w_l1 * l1_loss + depth_loss + self.w_consistency * consistency_loss
        loss_details = {
            'pointmap_l1': l1_loss.item(),
            'depth_loss': depth_loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss
        }
        return total_loss, loss_details

def get_thermal_transforms(thermal_mean, thermal_std):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Normalize(mean=thermal_mean, std=thermal_std)
    ])

def main(args):
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    print("Creating dataset...")
    raw_dataset = FreiburgTrainThermalDataset(
        root_dir=config['data']['train_path'],
        annotations_path=config['data']['annotations_path'],
        transform=None,
        img_size=(config['data']['img_width'], config['data']['img_height']),
        show_progress=True
    )
    
    if config.get('data', {}).get('compute_stats', True):
        thermal_mean, thermal_std = compute_thermal_stats(raw_dataset)
        print(f"Computed thermal stats - Mean: {thermal_mean}, Std: {thermal_std}")
    else:
        thermal_mean = config.get('data', {}).get('mean', [0.5, 0.5, 0.5])
        thermal_std = config.get('data', {}).get('std', [0.5, 0.5, 0.5])
    
    # Lighten data augmentation to reduce training complexity
    augmentation = ThermalAugmentation(
        brightness=0.1,  # was 0.2
        contrast=0.1,    # was 0.2
        noise=0.02,      # was 0.05
        flip_prob=0.3    # was 0.5
    )
    
    print("Reading dataset...")
    with tqdm(total=1, desc="Loading dataset") as pbar:
        full_dataset = FreiburgTrainThermalDataset(
            root_dir=config['data']['train_path'],
            annotations_path=config['data']['annotations_path'],
            transform=get_thermal_transforms(thermal_mean, thermal_std),
            img_size=(config['data']['img_width'], config['data']['img_height']),
            show_progress=True,
            augmentation=augmentation
        )
    print("Done Reading dataset")
    
    
    total_samples = len(full_dataset)
    val_size = int(total_samples * config['training'].get('val_split', 0.2))
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    if args.debug:
        batch = next(iter(train_loader))
        print("Keys in batch:", batch.keys())
        print("img1 shape:", batch['img1'].shape)
        print("img2 shape:", batch['img2'].shape)
        print("gt_pointmap1 shape:", batch['gt_pointmap1'].shape)
        print("img1 min, max:", batch['img1'].min().item(), batch['img1'].max().item())
        print("img2 min, max:", batch['img2'].min().item(), batch['img2'].max().item())
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.title("Thermal Image 1")
        plt.imshow(batch['img1'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.subplot(2, 3, 2)
        plt.title("GT Depth 1")
        plt.imshow(batch['gt_pointmap1'][0, 2].cpu().numpy(), cmap='viridis')
        plt.subplot(2, 3, 3)
        plt.title("GT Pointmap 1 (XY)")
        plt.imshow(batch['gt_pointmap1'][0, 2].cpu().numpy(), cmap='viridis')
        plt.subplot(2, 3, 4)
        plt.title("Thermal Image 2")
        plt.imshow(batch['img2'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.subplot(2, 3, 5)
        plt.title("GT Depth 2")
        plt.imshow(batch['gt_pointmap2'][0, 2].cpu().numpy(), cmap='viridis')
        plt.subplot(2, 3, 6)
        plt.title("GT Pointmap 2 (XY)")
        plt.imshow(batch['gt_pointmap2'][0, 2].cpu().numpy(), cmap='viridis')
        plt.tight_layout()
        debug_path = os.path.join(args.checkpoint_dir, 'debug_batch.png')
        plt.savefig(debug_path)
        print(f"Debug visualization saved to {debug_path}")
    
    print("Creating model...")
    model_config = config.get('model', {})
    model = DUSt3R()
    model = model.to(device)
    model.enable_gradient_checkpointing()

    scaler = torch.cuda.amp.GradScaler()
    
    point_map_loss = PointMapLoss(
        w_l1=config['training'].get('w_l1', 1.0),
        w_consistency=config['training'].get('w_consistency', 0.5)
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    lr_init = float(config['training']['lr_init'])
    lr_min  = float(config['training']['lr_min'])
    total_epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup from 0.1*lr_init to lr_init
            return 0.1 + 0.9 * ((epoch + 1) / warmup_epochs)
        else:
            # Cosine decay from lr_init to lr_min
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return (lr_min + (lr_init - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))) / lr_init

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'scaler' in checkpoint: 
            scaler.load_state_dict(checkpoint['scaler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    print("Starting training...")
    best_val_metrics = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=point_map_loss,
            device=device,
            epoch=epoch,
            writer=writer,
            config=config,
            scaler=scaler
        )
        
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=point_map_loss,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        if (epoch + 1) % config['training']['save_freq'] == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'config': config,
                'val_metrics': val_metrics
            }, ckpt_path)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'config': config,
                'val_metrics': val_metrics
            }, os.path.join(args.checkpoint_dir, 'latest.pth'))
            
            if val_metrics['rmse'] < best_val_metrics:
                best_val_metrics = val_metrics['rmse']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'val_metrics': val_metrics
                }, os.path.join(args.checkpoint_dir, 'best.pth'))
                print(f"Saved new best model with RMSE: {best_val_metrics:.4f}")

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, config, scaler, accum_steps=4):
    model.train()
    running_loss = 0.0
    running_loss_components = {
        'pointmap_l1': 0.0,
        'depth_loss': 0.0,
        'consistency_loss': 0.0
    }

    grad_clip_value = config['training'].get('grad_clip', 1.0)
    
    optimizer.zero_grad()  # Zero gradients before starting accumulation
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        gt_pointmap1 = batch['gt_pointmap1'].to(device)
        gt_pointmap2 = batch['gt_pointmap2'].to(device)
        pose1 = batch['pose1'].to(device) if 'pose1' in batch else None
        pose2 = batch['pose2'].to(device) if 'pose2' in batch else None
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(img1, img2)
            pred_pointmap1 = outputs['pointmap1']
            pred_pointmap2 = outputs['pointmap2']

            if pred_pointmap1.shape[2:] != gt_pointmap1.shape[2:]:
                pred_pointmap1 = F.interpolate(pred_pointmap1, size=gt_pointmap1.shape[2:], mode='bilinear', align_corners=False)
                pred_pointmap2 = F.interpolate(pred_pointmap2, size=gt_pointmap2.shape[2:], mode='bilinear', align_corners=False)
            
            loss, loss_details = criterion(
                pred_pointmap1, pred_pointmap2, gt_pointmap1, gt_pointmap2, 
                img1, img2, pose1, pose2
            )
            loss = loss / accum_steps  # Normalize loss by accumulation steps
        
        scaler.scale(loss).backward()
        
        # Accumulate gradients: update only every accum_steps mini-batches
        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients for next accumulation
        
        running_loss += loss.item() * accum_steps  # Multiply back the loss for reporting
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('train/loss', loss.item() * accum_steps, global_step)
        for k, v in loss_details.items():
            writer.add_scalar(f'train/{k}', v, global_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
        
        if i % config['training'].get('log_image_freq', 100) == 0:
            img_grid = make_grid(img1[:4], normalize=True, scale_each=True)
            writer.add_image('train/img1', img_grid, global_step)
            pred_depth = pred_pointmap1[:4, 2:3]
            gt_depth = gt_pointmap1[:4, 2:3]
            pred_depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
            gt_depth_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
            writer.add_image('train/pred_depth', make_grid(pred_depth_norm, normalize=False), global_step)
            writer.add_image('train/gt_depth', make_grid(gt_depth_norm, normalize=False), global_step)
            error_map = torch.abs(pred_depth - gt_depth)
            error_map = error_map / (error_map.max() + 1e-8)
            writer.add_image('train/depth_error', make_grid(error_map, normalize=False), global_step)
        
        if i % config['training']['print_freq'] == 0:
            print(f"Epoch {epoch}, Iter {i}/{len(train_loader)}, Loss: {loss.item() * accum_steps:.4f}")
    
    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    # Log average sub-losses
    for k, v in running_loss_components.items():
        writer.add_scalar(f'train/epoch_{k}', v / len(train_loader), epoch)
    
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    running_metrics = {
        'rmse': 0.0,
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'acc_1.25': 0.0,
        'acc_1.25^2': 0.0,
        'acc_1.25^3': 0.0
    }
    
    val_images_logged = False
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            gt_pointmap1 = batch['gt_pointmap1'].to(device)
            gt_pointmap2 = batch['gt_pointmap2'].to(device)
            gt_depth = batch['gt_depth'].to(device)
            pose1 = batch['pose1'].to(device) if 'pose1' in batch else None
            pose2 = batch['pose2'].to(device) if 'pose2' in batch else None
            
            outputs = model(img1, img2)
            pred_pointmap1 = outputs['pointmap1']
            pred_pointmap2 = outputs['pointmap2']
            
            if pred_pointmap1.shape[2:] != gt_pointmap1.shape[2:]:
                pred_pointmap1 = F.interpolate(pred_pointmap1, size=gt_pointmap1.shape[2:], mode='bilinear', align_corners=False)
                pred_pointmap2 = F.interpolate(pred_pointmap2, size=gt_pointmap2.shape[2:], mode='bilinear', align_corners=False)
            
            loss, _ = criterion(
                pred_pointmap1, pred_pointmap2, gt_pointmap1, gt_pointmap2,
                img1, img2, pose1, pose2
            )
            running_loss += loss.item()
            
            # Extract depth from predicted pointmap (channel index 2)
            pred_depth = pred_pointmap1[:, 2:3]
            # Scale factor to align predicted depth to GT scale
            scale_factor = (gt_depth * pred_depth).sum() / ((pred_depth ** 2).sum() + 1e-8)
            pred_depth_aligned = pred_depth * scale_factor
            
            metrics = compute_depth_metrics(pred_depth_aligned, gt_depth)
            for k, v in metrics.items():
                running_metrics[k] += v
            
            if not val_images_logged and i == 0:
                pred_depth_viz = (pred_depth_aligned - pred_depth_aligned.min()) / (pred_depth_aligned.max() - pred_depth_aligned.min() + 1e-8)
                writer.add_image('val/img1', make_grid(img1, normalize=True, scale_each=True), epoch)
                writer.add_image('val/gt_depth', make_grid(gt_depth, normalize=True, scale_each=True), epoch)
                writer.add_image('val/pred_depth', make_grid(pred_depth_viz, normalize=False), epoch)
                val_images_logged = True
    
    num_batches = len(val_loader)
    avg_loss = running_loss / num_batches
    for k in running_metrics:
        running_metrics[k] /= num_batches
        writer.add_scalar(f'val/{k}', running_metrics[k], epoch)
    
    print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    print("Validation Metrics:")
    for k, v in running_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return running_metrics

if __name__ == "__main__":
    args = parse_args()
    main(args)