#!/usr/bin/env python3
import argparse
import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import MASt3r model and helper functions
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from utils.helpers import get_pointmap, compute_relative_pose_from_pointmaps

# Import your dataset class
from data.freiburg_dataset import FreiburgThermalDataset

def main(args):
    device = torch.device(args.device)
    
    # Create DataLoader with a custom collate_fn so that we get a list of samples
    dataset = FreiburgThermalDataset(root_dir=args.data_path, img_size=tuple(args.img_size))
    dataloader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=2, 
                                               shuffle=False, 
                                               num_workers=args.num_workers,
                                               collate_fn=lambda x: x)
    
    # Load the pretrained MASt3r model.
    model = AsymmetricMASt3R.from_pretrained(args.ma_model_path)
    model = model.to(device)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    # Loop over the dataloader and process batches of 2 images
    for batch in tqdm(dataloader, desc="Verifying annotations"):
        # Each batch is now a list of dictionaries
        if not batch or len(batch) < 2:
            continue
        
        try:
            # Collect the images and paths from the two items
            images = torch.stack([item['rgb'] for item in batch], dim=0)  # shape: [2, 3, H, W]
            paths = [item['rgb_path'] for item in batch]
        except Exception as e:
            print(f"Error collating batch: {e}")
            continue
        
        if images.shape[0] < 2:
            continue
        
        img1 = images[0].to(device)
        img2 = images[1].to(device)
        
        # Ensure images are normalized to [0,1]
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
        
        # Create a dummy instance tensor for MASt3r
        dummy_instance = torch.zeros((1, 1, img1.shape[-2], img1.shape[-1]), device=device)
        
        # Build the image pair for inference
        image_pair = [(
            {"img": img1.unsqueeze(0), "instance": dummy_instance},
            {"img": img2.unsqueeze(0), "instance": dummy_instance}
        )]
        
        # Run inference through MASt3r to generate pointmaps
        out = inference(image_pair, model, device, batch_size=1, verbose=False)
        if 'pred1' not in out or 'pred2' not in out:
            print("Inference failed for this pair.")
            continue
        
        pm1 = get_pointmap(out['pred1'])  # shape: [1, 3, H, W]
        pm2 = get_pointmap(out['pred2'])  # shape: [1, 3, H, W]
        pm1_np = pm1[0].cpu().numpy()      # [3, H, W]
        pm2_np = pm2[0].cpu().numpy()      # [3, H, W]
        depth_value_1 = pm1_np[..., 2]       # Z channel from first pointmap
        
        # Compute relative pose for verification
        relative_pose = compute_relative_pose_from_pointmaps(pm1_np, pm2_np)
        pose1 = np.eye(4)
        
        # Build annotation dictionary (for verification only)
        annotation = {
            'pose1': pose1,
            'pose2': relative_pose,
            'depth_value_1': depth_value_1,
            'intrinsics': None,  # Not used here
            'pointmap1': pm1_np,
            'pointmap2': pm2_np,
            'frame1_path': paths[0],
            'frame2_path': paths[1]
        }
        
        # Save the visualization to disk
        save_visualization(annotation, args.output_dir, count)
        
        count += 1
        if count >= args.max_images:
            break

def save_visualization(annotation, output_dir, index):
    """
    Saves a visualization image containing:
      - Frame 1 (from frame1_path)
      - Frame 2 (from frame2_path)
      - Normalized depth map (from depth_value_1)
    """
    frame1_path = annotation.get('frame1_path')
    frame2_path = annotation.get('frame2_path')
    
    # Load images
    if frame1_path and os.path.exists(frame1_path):
        img1 = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    else:
        img1 = None
    if frame2_path and os.path.exists(frame2_path):
        img2 = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:
        img2 = None
    
    # Get depth map from depth_value_1
    depth = annotation.get('depth_value_1')
    if depth is None:
        print("No depth information in annotation.")
        return
    depth = np.array(depth)
    valid = depth > 0
    if valid.sum() > 0:
        dmin = np.percentile(depth[valid], 5)
        dmax = np.percentile(depth[valid], 95)
    else:
        dmin, dmax = depth.min(), depth.max()
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    if img1 is not None:
        axs[0].imshow(img1)
        axs[0].set_title("Frame 1")
    else:
        axs[0].text(0.5, 0.5, "Frame 1 not available", horizontalalignment='center', verticalalignment='center')
    axs[0].axis('off')
    
    if img2 is not None:
        axs[1].imshow(img2)
        axs[1].set_title("Frame 2")
    else:
        axs[1].text(0.5, 0.5, "Frame 2 not available", horizontalalignment='center', verticalalignment='center')
    axs[1].axis('off')
    
    im = axs[2].imshow(depth_norm, cmap='viridis')
    axs[2].set_title("Depth from Frame 1")
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2])
    
    filename = os.path.join(output_dir, f"verification_{index:04d}.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved visualization: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and visualize re-annotated image pairs")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--ma_model_path', type=str, required=True, help='Path to the MASt3r pretrained checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224,224], help='Image size (width height)')
    parser.add_argument('--max_images', type=int, default=20, help='Max number of image pairs to verify')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the verification images')
    args = parser.parse_args()
    main(args)