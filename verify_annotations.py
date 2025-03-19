#!/usr/bin/env python3
import argparse
import os
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_annotation(ann_path):
    """Load an annotation file (assumed to be a .npy file with a dict)."""
    try:
        annotation = np.load(ann_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {ann_path}: {e}")
        annotation = None
    return annotation

def check_annotation(annotation, ann_path):
    """Verify that the annotation contains required keys and valid shapes."""
    required_keys = ['frame1_path', 'pointmap1', 'pointmap2', 
                     'depth_value_1', 'intrinsics', 'pose1', 'pose2']
    missing = [key for key in required_keys if key not in annotation]
    if missing:
        print(f"Annotation {os.path.basename(ann_path)} missing keys: {missing}")
        return False
    valid = True
    # For pointmaps: expected shape can be either (3, H, W) or (H, W, 3)
    for key in ['pointmap1', 'pointmap2']:
        arr = annotation[key]
        if arr.ndim != 3:
            print(f"Annotation {os.path.basename(ann_path)}: key {key} has invalid number of dimensions: {arr.ndim}")
            valid = False
        elif not (arr.shape[0] == 3 or arr.shape[-1] == 3):
            print(f"Annotation {os.path.basename(ann_path)}: key {key} has invalid shape {arr.shape} (expected a 3-channel image)")
            valid = False

    if annotation['depth_value_1'].ndim != 2:
        print(f"Annotation {os.path.basename(ann_path)}: depth_value_1 shape {annotation['depth_value_1'].shape} (expected 2D)")
        valid = False
    if annotation['intrinsics'].shape != (3, 3):
        print(f"Annotation {os.path.basename(ann_path)}: intrinsics shape {annotation['intrinsics'].shape} (expected (3,3))")
        valid = False
    for key in ['pose1', 'pose2']:
        if annotation[key].shape != (4, 4):
            print(f"Annotation {os.path.basename(ann_path)}: {key} shape {annotation[key].shape} (expected (4,4))")
            valid = False
    return valid

def print_annotation_stats(annotation, ann_path):
    """Print basic statistics for an annotation file."""
    print(f"\nAnnotation: {os.path.basename(ann_path)}")
    for key, value in annotation.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}, min {np.min(value):.4f}, max {np.max(value):.4f}, mean {np.mean(value):.4f}")
        else:
            print(f"  {key}: {value}")

def plot_annotation(annotation, ann_path, save_dir=None):
    """
    Plot the following for visual inspection:
      - The thermal image: by loading the 'frame1_path' and converting from fl_rgb to fl_ir_aligned if needed.
      - The depth channel from pointmap1.
      - The intrinsics and pose matrices printed as text.
    """
    # Convert RGB frame path to thermal path
    rgb_frame_path = annotation['frame1_path']
    thermal_path = rgb_frame_path.replace('fl_rgb', 'fl_ir_aligned')
    if not os.path.exists(thermal_path):
        print(f"Thermal image {thermal_path} not found.")
        return

    # Load the thermal image using cv2 (assume 16-bit image)
    thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
    if thermal_img is None:
        print(f"Failed to load thermal image from {thermal_path}")
        return
    thermal_img = thermal_img.astype(np.float32)
    thermal_img_norm = thermal_img / 65535.0  # Normalize

    # Get depth channel from pointmap1 (assume index 2)
    pointmap1 = annotation['pointmap1']
    depth_channel = pointmap1[2]  # shape: [H, W]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(thermal_img_norm, cmap='gray')
    axes[0].set_title("Thermal Image")
    axes[0].axis("off")
    im = axes[1].imshow(depth_channel, cmap='viridis')
    axes[1].set_title("GT Depth from Pointmap1 (Channel 2)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.suptitle(os.path.basename(ann_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(ann_path).replace('.npy', '.png'))
        plt.savefig(save_path)
        print(f"Saved annotation visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def main(args):
    ann_files = sorted(glob.glob(os.path.join(args.annotations_dir, "*.npy")))
    if not ann_files:
        print("No annotation files found in the directory.")
        return

    valid_count = 0
    total_depths = []
    for ann_path in tqdm(ann_files, desc="Verifying annotations"):
        annotation = load_annotation(ann_path)
        if annotation is None:
            continue
        valid = check_annotation(annotation, ann_path)
        if valid:
            valid_count += 1
            # Collect some depth statistics from depth_value_1
            depth = annotation['depth_value_1']
            pointmap1 = annotation['pointmap1']
            total_depths.append(depth.flatten())
            # Optionally, print out statistics for a few files:
            # if args.verbose:
            #     print_annotation_stats(annotation, ann_path)
            # Suppose pointmap1 is shape (3, H, W) after transposing if needed
            if pointmap1.shape[0] == 3:
                print("of shapen pointmap1.shape[0]")
                # Already channels-first: shape (3, H, W)
                z_channel = pointmap1[2]
            elif pointmap1.shape[-1] == 3:
                print("of shapen pointmap1.shape[-1]")
                # Channels-last: shape (H, W, 3)
                z_channel = pointmap1[..., 2]
            else:
                raise ValueError("Unexpected pointmap shape:", pointmap1.shape)
            depth_val = depth         # [H, W] from annotation

            # Print some stats
            print("Pointmap Z-channel: min =", z_channel.min(), 
                "max =", z_channel.max(), 
                "mean =", z_channel.mean())
            print("Depth Value from annotation: min =", depth_val.min(), 
                "max =", depth_val.max(), 
                "mean =", depth_val.mean())

            # Optional: measure difference
            diff = z_channel - depth_val
            print("Difference stats: min =", diff.min(), 
                "max =", diff.max(), 
                "mean =", diff.mean())
    print(f"\nFound {valid_count} valid annotations out of {len(ann_files)} files.")
    
    # Compute global depth statistics across all annotations
    if total_depths:
        all_depths = np.concatenate(total_depths)
        print("\nGlobal Depth Statistics:")
        print(f"  Min: {np.min(all_depths):.4f}, Max: {np.max(all_depths):.4f}, Mean: {np.mean(all_depths):.4f}, Std: {np.std(all_depths):.4f}")

    # Visualize a few random annotations if requested
    if args.visualize:
        import random
        sample_files = random.sample(ann_files, min(args.num_visualize, len(ann_files)))
        for ann_path in sample_files:
            annotation = load_annotation(ann_path)
            if annotation is not None and check_annotation(annotation, ann_path):
                plot_annotation(annotation, ann_path, save_dir=args.save_visualizations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and visualize pseudo-GT annotations for DUSt3R")
    parser.add_argument("--annotations_dir", type=str, required=True,
                        help="Path to the directory containing annotation .npy files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics for each annotation")
    parser.add_argument("--visualize", action="store_true", help="Visualize a few annotations")
    parser.add_argument("--num_visualize", type=int, default=5, help="Number of annotations to visualize")
    parser.add_argument("--save_visualizations", type=str, default=None,
                        help="Directory to save the visualizations instead of showing them")
    args = parser.parse_args()
    main(args)