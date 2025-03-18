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

# Disable the symmetrization check to avoid ambiguous tensor comparisons.
from dust3r.utils import misc
misc.is_symmetrized = lambda x, y: False

from models.dust3r import DUSt3R
from data.freiburg_train_dataset import FreiburgTrainThermalDataset
from utils.metrics import compute_depth_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train DUSt3R on thermal data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def get_thermal_transforms(config):
    """
    Define transformations specific for thermal images
    """
    from torchvision import transforms
    
    # Thermal-specific normalization values
    thermal_mean = [0.5, 0.5, 0.5]  # For 3-channel thermal images
    thermal_std = [0.5, 0.5, 0.5]
    
    return transforms.Compose([
        transforms.Normalize(mean=thermal_mean, std=thermal_std)
    ])

def main(args):
    # Load config
    print("start main")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config['device'])
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    # Create dataset with thermal-specific transforms
    print("Reading dataset...")
    with tqdm(total=1, desc="Loading dataset") as pbar:
        full_dataset = FreiburgTrainThermalDataset(
            root_dir=config['data']['train_path'],
            annotations_path=config['data']['annotations_path'],
            transform=get_thermal_transforms(config),
            img_size=(config['data']['img_width'], config['data']['img_height']),
            show_progress=True
        )
        pbar.update(1)
    print("Done Reading dataset")


    
    
    # Split dataset into training and validation
    total_samples = len(full_dataset)
    val_size = int(total_samples * config['training'].get('val_split', 0.2))
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    batch = next(iter(train_loader))
    print("Keys in batch:", batch.keys())
    print("img1 shape:", batch['img1'].shape)  # Expected: [B, 3, H, W]
    print("img2 shape:", batch['img2'].shape)  # Expected: [B, 3, H, W]

    # Check value ranges (for normalization, etc.)
    print("img1 min, max:", batch['img1'].min().item(), batch['img1'].max().item())
    print("img2 min, max:", batch['img2'].min().item(), batch['img2'].max().item())
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    print("Done creating train and val loader for dataset")

    
    # Create model (use small variant as specified in requirements)
    model = DUSt3R()  # Specify the small variant
    model = model.to(device)
    
    # Rest of your training code remains the same...
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create lr scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("starting the  training")

    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            config=config
        )
        
        # Validate
        validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint for this epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, ckpt_path)
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(args.checkpoint_dir, 'latest.pth'))

def train_one_epoch(model, train_loader, optimizer, device, epoch, writer, config):
    model.train()
    running_loss = 0.0
    
    pointmap_criterion = nn.L1Loss()
    
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        # Move data to device
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        gt_pointmap1 = batch['gt_pointmap1'].to(device)
        gt_pointmap2 = batch['gt_pointmap2'].to(device)
        
        # Forward pass
        outputs = model(img1, img2)
        pred_pointmap1 = outputs['pointmap1']
        pred_pointmap2 = outputs['pointmap2']
        
        # Compute loss
        loss_pointmap1 = pointmap_criterion(pred_pointmap1, gt_pointmap1)
        loss_pointmap2 = pointmap_criterion(pred_pointmap2, gt_pointmap2)
        
        # Total loss
        loss = loss_pointmap1 + loss_pointmap2
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log scalar losses
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/loss_pointmap1', loss_pointmap1.item(), global_step)
        writer.add_scalar('train/loss_pointmap2', loss_pointmap2.item(), global_step)
        
        # Optionally log images every N iterations
        if i % config['training'].get('log_image_freq', 100) == 0:
            # Create a grid of input images
            img_grid = make_grid(img1, normalize=True, scale_each=True)
            writer.add_image('train/img1', img_grid, global_step)
            
            # Log predicted and ground truth pointmaps (visualizing one channel, e.g. the depth channel at index 2)
            pred_depth = pred_pointmap1[:, 2:3, :, :]  # shape [B,1,H,W]
            gt_depth = gt_pointmap1[:, 2:3, :, :]
            writer.add_image('train/pred_depth', make_grid(pred_depth, normalize=True, scale_each=True), global_step)
            writer.add_image('train/gt_depth', make_grid(gt_depth, normalize=True, scale_each=True), global_step)
        
        if i % config['training']['print_freq'] == 0:
            print(f"Epoch {epoch}, Iter {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

def validate(model, val_loader, device, epoch, writer):
    model.eval()
    running_metrics = {
        'rmse': 0.0,
        'abs_rel': 0.0,
        'acc_1.25': 0.0,
        'acc_1.25^2': 0.0,
        'acc_1.25^3': 0.0
    }
    
    # For image logging during validation, we take the first batch
    val_images_logged = False
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            gt_depth = batch['gt_depth'].to(device)
            
            outputs = model(img1, img2)
            pred_pointmap1 = outputs['pointmap1']
            pred_depth = pred_pointmap1[:, 2, :, :].unsqueeze(1)  # add channel dimension
            
            metrics = compute_depth_metrics(pred_depth, gt_depth)
            for k, v in metrics.items():
                running_metrics[k] += v
            
            # Log images for the first batch of validation
            if not val_images_logged:
                writer.add_image('val/img1', make_grid(img1, normalize=True, scale_each=True), epoch)
                writer.add_image('val/gt_depth', make_grid(gt_depth, normalize=True, scale_each=True), epoch)
                writer.add_image('val/pred_depth', make_grid(pred_depth, normalize=True, scale_each=True), epoch)
                val_images_logged = True
    
    # Average metrics over validation batches
    for k in running_metrics:
        running_metrics[k] /= len(val_loader)
        writer.add_scalar(f'val/{k}', running_metrics[k], epoch)
    
    print(f"Validation Metrics (Epoch {epoch}):")
    for k, v in running_metrics.items():
        print(f"  {k}: {v:.4f}")

def get_transforms(config):
    from torchvision import transforms
    # Define transforms for both train and validation
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

if __name__ == "__main__":
    args = parse_args()
    main(args)