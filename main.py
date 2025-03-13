import sys
sys.path.append('/home/user/victorv1/Cuda/3d-vision-thermal-images/mast3r/dust3r/croco')

# main.py
import argparse
import os
import yaml
from pathlib import Path
import glob
import numpy as np
import torch
from utils.visualization import visualize_depth_map


def parse_args():
    parser = argparse.ArgumentParser(description='Thermal 3D Vision Project')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['generate_annotations', 'visualise', 'train', 'evaluate', 'evaluate_ais'], 
                       help='Mode to run')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for evaluation or resuming training')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    if args.mode == 'visualise':
        ANNOTATION_DIR = "data/annotations"
        NUM_TO_VISUALIZE = 10

        annotation_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.npy")))[:NUM_TO_VISUALIZE]
        print(len(annotation_files))
        for i, ann_file in enumerate(annotation_files):
            annotation = np.load(ann_file, allow_pickle=True).item()
            depth = annotation["depth1"]
            save_path = os.path.join("data/visualisation", f"depth_map_{i}.png")
            visualize_depth_map(depth, save_path=save_path)

    if args.mode == 'generate_annotations':
        # Generate pseudo-GT annotations
        from scripts.generate_annotations import main as generate_annotations_main
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        generate_args = argparse.Namespace(
            data_path=config['data']['train_path'],
            output_path="data/annotations",
            calib_yaml="config/calibrations/thermal_stereo_calib.yaml",
            batch_size=8,
            num_workers=4,
            device=device
        )
        generate_annotations_main(generate_args)
        
    elif args.mode == 'train':
        # Train the model
        from scripts.train import main as train_main
        train_args = argparse.Namespace(
            config=args.config,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
            resume=args.checkpoint
        )
        train_main(train_args)
        
    elif args.mode == 'evaluate':
        # Evaluate on Freiburg test set
        from scripts.evaluate import main as evaluate_main
        evaluate_args = argparse.Namespace(
            config=args.config,
            checkpoint=args.checkpoint,
            output_dir=os.path.join(args.output_dir, 'freiburg_evaluation'),
            vis_samples=10
        )
        evaluate_main(evaluate_args)
        
    elif args.mode == 'evaluate_ais':
        # Evaluate on AIS test data
        from scripts.evaluate_ais import main as evaluate_ais_main
        evaluate_ais_args = argparse.Namespace(
            config=args.config,
            checkpoint=args.checkpoint,
            output_dir=os.path.join(args.output_dir, 'ais_evaluation'),
            vis_samples=10
        )
        evaluate_ais_main(evaluate_ais_args)

if __name__ == "__main__":
    args = parse_args()
    main(args)