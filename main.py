import sys
import os
sys.path.append('/home/user/victorv1/Cuda/3d-vision-thermal-images/mast3r/dust3r/croco')
sys.path.append(os.path.join(os.path.dirname(__file__), 'mast3r'))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# main.py
import argparse
import os
import yaml
from pathlib import Path
import glob
import numpy as np
import torch
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Thermal 3D Vision Project')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate_annotations', 'visualise', 'verify_annotations', 'train', 'evaluate', 'evaluate_ais', 'evaluate_combined'],
                        help='Mode to run')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for evaluation or resuming training')
    # New arguments for combined evaluation:
    parser.add_argument('--ft_checkpoint', type=str, default='/home/nfs/inf6/data/cudalab/victorv1/results/checkpoints/checkpoint_epoch_139.pth',
                        help='Path to fine-tuned DUSt3R checkpoint (for evaluate_combined)')
    parser.add_argument('--base_checkpoint', type=str, default='checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth',
                        help='Path to base DUSt3R checkpoint (for evaluate_combined)')
    parser.add_argument('--ma_checkpoint', type=str, default='checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                        help='Path to MASt3R checkpoint (for evaluate_combined)')
    parser.add_argument('--output_dir', type=str, default='/home/nfs/inf6/data/cudalab/victorv1/results', help='Directory to save results')
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


    if args.mode == 'generate_annotations':
        # Generate pseudo-GT annotations
        from scripts.generate_annotations import main as generate_annotations_main
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        generate_args = argparse.Namespace(
            data_path=config['data']['train_path'],
            output_path="data/annotations",
            calib_yaml="config/calibrations/thermal_stereo_calib.yaml",
            batch_size=2,
            num_workers=4,
            device=device
        )
        generate_annotations_main(generate_args)
    
    elif args.mode == 'verify_annotations':
        # verify annotations
        from scripts.verify_annotations import main as verify_annotations
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        verify_annotations_args = argparse.Namespace(
            data_path=config['data']['train_path'],
            ma_model_path="/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            output_dir=config['data']['verify_annotations_path'],
            device=device,
            img_size=[224,224],
            num_workers=4
        )
        if verify_annotations_args.output_dir is not None:
            os.makedirs(verify_annotations_args.output_dir, exist_ok=True)
        verify_annotations(verify_annotations_args)
        
    elif args.mode == 'train':
        # Train the model
        from scripts.train import main as train_main
        train_args = argparse.Namespace(
            config=args.config,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
            resume=args.checkpoint,
            debug=True
        )
        train_main(train_args)
        
    elif args.mode == 'evaluate':
        # Evaluate on Freiburg test set
        from scripts.evaluate import main as evaluate_main
        evaluate_args = argparse.Namespace(
            config=args.config,
            ft_checkpoint=args.ft_checkpoint,
            base_checkpoint=args.base_checkpoint,
            calib_yaml="config/calibrations/thermal_stereo_calib.yaml",
            ma_checkpoint=args.ma_checkpoint,
            output_dir=os.path.join(args.output_dir, 'eval_results'),
            vis_samples=50
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