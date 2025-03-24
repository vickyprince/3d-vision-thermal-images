#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.visualization import visualize_annotation_correctness
from utils.helpers import get_pointmap, compute_relative_pose_from_pointmaps, custom_collate, get_intrinsics_from_yaml, load_calibrations

def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo-GT with relative pose, depth, intrinsics.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to your dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Where to save annotations')
    parser.add_argument('--calib_yaml', type=str, required=True, help='YAML with camera intrinsics')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main(args):
    # Load calibration
    calib = load_calibrations(args.calib_yaml)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create a directory for visualizations
    viz_dir = os.path.join(args.output_path, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load pretrained MASt3r model
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    model = AsymmetricMASt3R.from_pretrained('checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    model = model.to(args.device)
    model.eval()

    # Create  dataset & dataloader 
    from data.freiburg_dataset import FreiburgThermalDataset
    from utils.helpers import custom_collate
    dataset = FreiburgThermalDataset(args.data_path)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=custom_collate)

    # Iterate over the data
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating Annotations")):
            if not batch or 'rgb' not in batch:
                continue

            images = batch['rgb']  # [B, 3, H, W]
            paths = batch.get('rgb_path', [])
            thermal_paths = batch.get('thermal_path', [])
            # Make sure we have at least two rgb images and one thermal image
            if len(paths) < 2 or len(thermal_paths) < 1:
                continue

            img1 = images[0].to(args.device)
            img2 = images[1].to(args.device)
            base_name1 = os.path.splitext(os.path.basename(paths[0]))[0]
            ann_out_path = os.path.join(args.output_path, f"{base_name1}.npy")
            if os.path.exists(ann_out_path):
                print(f"Skipping existing annotation: {ann_out_path}")
                continue
            else:

                # If your images are in [0..255], scale them to [0..1]
                if img1.max() > 1.0:
                    img1 = img1 / 255.0
                if img2.max() > 1.0:
                    img2 = img2 / 255.0

                # Build pair for MASt3r inference
                dummy_instance = torch.zeros((1,1,img1.shape[-2],img1.shape[-1]), device=args.device)
                image_pair = [(
                    {"img": img1.unsqueeze(0), "instance": dummy_instance},
                    {"img": img2.unsqueeze(0), "instance": dummy_instance}
                )]

                out = inference(image_pair, model, args.device, batch_size=1, verbose=False)
                if 'pred1' not in out or 'pred2' not in out:
                    continue

                pm1_dict = out['pred1']
                pm2_dict = out['pred2']

                pm1 = get_pointmap(pm1_dict)  # shape [1, 3, H, W]
                pm2 = get_pointmap(pm2_dict)  # shape [1, 3, H, W]

                # Convert to numpy
                pm1_np = pm1[0].cpu().numpy()  # [3, H, W]
                pm2_np = pm2[0].cpu().numpy()  # [3, H, W]

                depth_value_1 = pm1_np[..., 2]

                # Relative pose: assume first image is identity, compute second from pointmaps
                relative_pose_1_to_2 = compute_relative_pose_from_pointmaps(pm1_np, pm2_np)
                pose1 = np.eye(4)  # canonical

                # Intrinsics from calibration
                intrinsics = get_intrinsics_from_yaml(calib, paths[0])
                fx, fy, cx, cy = intrinsics
                intrinsics = np.array([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,   1 ]
                ])

                # Build and save annotation
                annotation = {
                    'pose1': pose1,                        # identity
                    'pose2': relative_pose_1_to_2,         # from pointmaps
                    'depth_value_1': depth_value_1,        # Z of first pointmap
                    'intrinsics': intrinsics,
                    'pointmap1': pm1_np,                   # optional
                    'pointmap2': pm2_np,                   # optional
                    'frame1_path': paths[0]                # for visualization title
                }
                np.save(ann_out_path, annotation)

            # Visualize annotation and save visualization image (do not show interactively)
            if batch_idx % 500 == 0:
                viz_path = os.path.join(viz_dir, f"{base_name1}_viz.png")
                visualize_annotation_correctness(annotation, paths[0], thermal_paths[0], save_path=viz_path)
                print(f"Saved visualization: {viz_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)