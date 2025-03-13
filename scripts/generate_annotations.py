#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import yaml  # for reading calibration YAML file

from torch.utils.data import DataLoader
from mast3r.model import AsymmetricMASt3R
from data.freiburg_dataset import FreiburgThermalDataset
from utils.camera_utils import compute_pose_from_pointmaps
from utils.visualization import visualize_depth_map
from dust3r.inference import inference
from utils.helpers import get_pointmap

def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo-GT annotations using MASt3R with YAML calibrations.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Freiburg dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save annotations')
    parser.add_argument('--calib_yaml', type=str, required=True, help='Path to calibration YAML file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (for DataLoader)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def pad_to_multiple(img, multiple=16):
    """
    Pads a 4D tensor (1, C, H, W) on the right and bottom so that its H and W become multiples of 'multiple'.
    Returns the padded tensor and the original (H, W) to allow later cropping.
    """
    _, _, h, w = img.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_bottom = new_h - h
    pad_right = new_w - w
    padded = F.pad(img, (0, pad_right, 0, pad_bottom))
    return padded, h, w

def load_calibrations(calib_yaml):
    """Load calibration data from a YAML file."""
    with open(calib_yaml, "r") as f:
        calib = yaml.safe_load(f)
    return calib

def get_calibration(calib, image_path):
    """
    Select calibration based on the image file name.
    If "fl_rgb" is in the filename, we assume left-camera calibration; otherwise, right.
    """
    base = os.path.basename(image_path).lower()
    if "fl_rgb" in base:
        intrinsics = np.array(calib["left"]["intrinsics"])
        pose = np.eye(4)
    else:
        intrinsics = np.array(calib["right"]["intrinsics"])
        # Use the provided transformation matrix if available
        pose = np.array(calib["right"].get("T_cn_cnm1", np.eye(4)))
    return intrinsics, pose

def main(args):
    # Create output directory
    print(f"Creating output directory: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load calibration from YAML file
    calib = load_calibrations(args.calib_yaml)
    
    # Initialize MASt3R model
    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained('checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    print(f"Moving model to device: {args.device}")
    device = args.device
    model = model.to(args.device)
    model.eval()
    print("Model ready")
    
    # Load dataset; FreiburgThermalDataset returns keys: 'img1', 'img2', 'frame1_path', 'frame2_path', and 'intrinsics'
    print(f"Loading dataset from: {args.data_path}")
    dataset = FreiburgThermalDataset(args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Generate annotations using inference()
    print("Starting annotation generation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch is None:
                continue

            # Inside your main loop:
            if 'rgb' in batch:
                images = batch['rgb']          # [B, 3, H, W]
                paths = batch.get('rgb_path', None)
                thermal_paths = batch.get('thermal_path', None)
            else:
                images = batch['img1']         # fallback if 'rgb' key is missing
                paths = batch.get('frame1_path', None)
                thermal_paths = batch.get('thermal_path', None)

            if paths is None or len(paths) == 0:
                print("Warning: No image path found, skipping.")
                continue

            if thermal_paths is None or len(thermal_paths) == 0 or not os.path.isfile(thermal_paths[0]):
                print("Warning: Thermal path missing for sample, skipping.")
                continue

            # Retrieve calibration parameters using the first image path.
            intrinsics, calib_pose = get_calibration(calib, paths[0])
            img_path = paths[0]

            # Duplicate image if batch contains one image.
            if len(images) == 1:
                img1 = images[0].to(device)
                img2 = img1.clone()
                base_name1 = os.path.splitext(os.path.basename(paths[0]))[0]
                base_name2 = base_name1 + "_dup"
            else:
                img1 = images[0].to(device)
                img2 = images[1].to(device)
                base_name1 = os.path.splitext(os.path.basename(paths[0]))[0]
                base_name2 = os.path.splitext(os.path.basename(paths[1]))[0]

            # Create a dummy "instance" tensor (dummy mask) that matches the image dimensions.
            dummy_instance = torch.zeros((1, 1, img1.shape[-2], img1.shape[-1]), device=device)

            # Build the image pair for inference.
            image_pair = [(
                {"img": img1.unsqueeze(0), "instance": dummy_instance},
                {"img": img2.unsqueeze(0), "instance": dummy_instance}
            )]
            
            torch.cuda.empty_cache()
            # Run inference.
            out = inference(image_pair, model, device, batch_size=1, verbose=False)

            # Retrieve predicted pointmaps using the helper 'get_pointmap'.
            if 'pred1' not in out or 'pred2' not in out:
                print(f"Inference output missing keys for sample {base_name1}, skipping.")
                continue
            pm1_dict = out.get("pred1", {})
            pm2_dict = out.get("pred2", {})
            if pm1_dict is None or pm2_dict is None:
                print(f"Pointmap keys missing for sample {base_name1}, skipping.")
                continue

            pointmap1 = get_pointmap(pm1_dict)  # expected shape [1, 3, H, W]
            pointmap2 = get_pointmap(pm2_dict)

            # Extract depth values (Z channel) from both pointmaps.
            depth_value_1 = pointmap1[0, 2, :, :].cpu().numpy()  # shape: [H, W]
            depth_value_2 = pointmap2[0, 2, :, :].cpu().numpy()

            # Build annotation dictionary with the desired keys.
            annotation = {
                'pointmap1': pointmap1[0].cpu().numpy(),  # [3, H, W]
                'pointmap2': pointmap2[0].cpu().numpy(),
                'depth_value_1': depth_value_1,
                'depth_value_2': depth_value_2,
                'pose': calib_pose,
                'K': intrinsics,
                'frame1_path': img_path
            }
            ann_id = f"{base_name1}"
            np.save(os.path.join(args.output_path, f"{ann_id}.npy"), annotation)

            # Optionally visualize the depth map from the first image.
            visualize_depth_map(
                depth_value_1,
                os.path.join(args.output_path, f"{ann_id}_depth.png")
            )
    
    print("Annotation generation complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)