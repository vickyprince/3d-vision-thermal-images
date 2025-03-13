# scripts/train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.dust3r import DUSt3R
from data.freiburg_dataset import FreiburgThermalDataset
from utils.metrics import compute_depth_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train DUSt3R on thermal data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config['device'])
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    # Create dataset and dataloader
    train_dataset = FreiburgThermalDataset(
        root_dir=config['data']['train_path'],
        annotations_path=config['data']['annotations_path'],
        transform=get_transforms(config, 'train')
    )
    
    val_dataset = FreiburgThermalDataset(
        root_dir=config['data']['val_path'],
        annotations_path=config['data']['annotations_path'],
        transform=get_transforms(config, 'val')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model = DUSt3R(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create lr scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
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
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
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
    
    for i, batch in enumerate(tqdm(train_loader)):
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
        
        # Update running loss
        running_loss += loss.item()
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/loss_pointmap1', loss_pointmap1.item(), global_step)
        writer.add_scalar('train/loss_pointmap2', loss_pointmap2.item(), global_step)
        
        # Print loss occasionally
        if i % config['training']['print_freq'] == 0:
            print(f"Epoch {epoch}, Iter {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Log average loss for epoch
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
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
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
            
            # Update running metrics
            for k, v in metrics.items():
                running_metrics[k] += v
    
    # Average metrics
    for k in running_metrics:
        running_metrics[k] /= len(val_loader)
        writer.add_scalar(f'val/{k}', running_metrics[k], epoch)
    
    print(f"Validation Metrics (Epoch {epoch}):")
    for k, v in running_metrics.items():
        print(f"  {k}: {v:.4f}")

def get_transforms(config, split):
    # Implement the appropriate transforms for training and validation
    # This is a placeholder - expand based on actual requirements
    from torchvision import transforms
    
    if split == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

if __name__ == "__main__":
    args = parse_args()
    main(args)