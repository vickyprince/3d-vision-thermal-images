#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.visualization import visualize_annotation_correctness
from utils.helpers import (
    get_pointmap, 
    compute_relative_pose_from_pointmaps, 
    custom_collate, 
    get_intrinsics_from_yaml, 
    load_calibrations
)

def parse_args():
    """
    Parse command-line arguments for generating pseudo-ground truth annotations.
    """
    parser = argparse.ArgumentParser(
        description='Generate pseudo-GT with relative pose, depth, and intrinsics.'
    )
    parser.add_argument('--data_path', type=str, required=True, help='Path to your dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save annotations')
    parser.add_argument('--calib_yaml', type=str, required=True, help='Path to YAML file with camera intrinsics')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use ("cuda" or "cpu")')
    return parser.parse_args()

def main(args):
    """
    Generate pseudo-ground truth annotations using the pretrained MASt3r model.
    Annotations are saved as .npy files and visualizations are saved periodically.
    """
    calib = load_calibrations(args.calib_yaml)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create a directory for visualizations.
    viz_dir = os.path.join(args.output_path, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load pretrained MASt3r model.
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    model = AsymmetricMASt3R.from_pretrained('checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    model = model.to(args.device)
    model.eval()

    # Load dataset and create a dataloader.
    from data.freiburg_dataset import FreiburgThermalDataset
    # from data.test_annotaation_dataset import FreiburgThermalTestDataset    

    dataset = FreiburgThermalDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    # Iterate over dataset batches and generate annotations.
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating Annotations")):
            if not batch or 'rgb' not in batch:
                continue

            images = batch['rgb']  # shape: [B, 3, H, W]
            paths = batch.get('rgb_path', [])
            thermal_paths = batch.get('thermal_path', [])
            if len(paths) < 2 or len(thermal_paths) < 1:
                if len(paths) < 2 and len(paths) == 1:
                    print("Only one RGB image found; duplicating the image to create a pair.")
                    images = torch.cat([images, images], dim=0)
                    paths = paths * 2  # Duplicate the path list
                else:
                    continue

            img1 = images[0].to(args.device)
            img2 = images[1].to(args.device)
            base_name1 = os.path.splitext(os.path.basename(paths[0]))[0]
            ann_out_path = os.path.join(args.output_path, f"{base_name1}.npy")
            if os.path.exists(ann_out_path):
                print(f"Skipping existing annotation: {ann_out_path}")
                continue
            else:
                # Normalize images if required.
                if img1.max() > 1.0:
                    img1 = img1 / 255.0
                if img2.max() > 1.0:
                    img2 = img2 / 255.0

                dummy_instance = torch.zeros((1, 1, img1.shape[-2], img1.shape[-1]), device=args.device)
                image_pair = [(
                    {"img": img1.unsqueeze(0), "instance": dummy_instance},
                    {"img": img2.unsqueeze(0), "instance": dummy_instance}
                )]

                out = inference(image_pair, model, args.device, batch_size=1, verbose=False)
                if 'pred1' not in out or 'pred2' not in out:
                    continue

                pm1 = get_pointmap(out['pred1'])  # shape: [1, 3, H, W]
                pm2 = get_pointmap(out['pred2'])  # shape: [1, 3, H, W]

                pm1_np = pm1[0].cpu().numpy()  # shape: [3, H, W]
                pm2_np = pm2[0].cpu().numpy()  # shape: [3, H, W]
                depth_value_1 = pm1_np[..., 2]

                relative_pose_1_to_2 = compute_relative_pose_from_pointmaps(pm1_np, pm2_np)
                pose1 = np.eye(4)

                intrinsics = get_intrinsics_from_yaml(calib, paths[0])
                fx, fy, cx, cy = intrinsics
                intrinsics = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0,  0,  1]])

                annotation = {
                    'pose1': pose1,
                    'pose2': relative_pose_1_to_2,
                    'depth_value_1': depth_value_1,
                    'intrinsics': intrinsics,
                    'pointmap1': pm1_np,
                    'pointmap2': pm2_np,
                    'frame1_path': paths[0]
                }
                np.save(ann_out_path, annotation)

            # Periodically visualize annotations.
            if batch_idx % 500 == 0:
                viz_path = os.path.join(viz_dir, f"{base_name1}_viz.png")
                visualize_annotation_correctness(annotation, paths[0], thermal_paths[0], save_path=viz_path)
                print(f"Saved visualization: {viz_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)