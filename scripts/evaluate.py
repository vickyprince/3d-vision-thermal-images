#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.helpers import get_pointmap, get_intrinsics_from_yaml, load_calibrations
from utils.visualization import visualize_evaluation, visualize_pointcloud
from utils.metrics import compute_depth_metrics

from models.dust3r import DUSt3R
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from data.freiburg_evaluate_dataset import FreiburgEvaluateThermalDataset

def parse_args():
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Compare fine-tuned DUSt3R vs. combined (base DUSt3R + MASt3R) for monocular depth estimation on thermal data"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ft_checkpoint", type=str, required=True, help="Path to fine-tuned DUSt3R checkpoint")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Path to base DUSt3R checkpoint")
    parser.add_argument("--ma_checkpoint", type=str, required=True, help="Path to MASt3R checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--vis_samples", type=int, default=10, help="Number of samples to visualize qualitatively")
    return parser.parse_args()

def get_transform(mean, std):
    """
    Create a normalization transform.
    """
    from torchvision import transforms
    return transforms.Compose([transforms.Normalize(mean=mean, std=std)])

def load_models(ft_ckpt, base_ckpt, ma_ckpt, device):
    """
    Load fine-tuned DUSt3R, base DUSt3R, and MASt3R models.
    """
    ft_model = DUSt3R().to(device)
    ft_state = torch.load(ft_ckpt, map_location=device)
    ft_model.load_state_dict(ft_state["model"])
    ft_model.eval()

    base_model = DUSt3R().to(device)
    base_state = torch.load(base_ckpt, map_location=device)
    base_model.load_state_dict(base_state["model"], strict=False)
    base_model.eval()

    ma_model = AsymmetricMASt3R.from_pretrained(ma_ckpt)
    ma_model = ma_model.to(device)
    ma_model.eval()

    return ft_model, base_model, ma_model

def main(args):
    """
    Run the evaluation loop for monocular depth estimation.
    """
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config.get("device", "cuda"))
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    mean = config.get("data", {}).get("mean", [0.5, 0.5, 0.5])
    std = config.get("data", {}).get("std", [0.5, 0.5, 0.5])
    transform = get_transform(mean, std)

    ft_model, base_model, ma_model = load_models(
        args.ft_checkpoint, args.base_checkpoint, args.ma_checkpoint, device
    )

    test_dataset = FreiburgEvaluateThermalDataset(
        root_dir=config["data"]["test_path"],
        transform=transform,
        img_size=(config["data"]["img_width"], config["data"]["img_height"])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    ref_scale = config.get("evaluation", {}).get("reference_scale", 10.0)
    ft_metrics = {"rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0, "acc_1.25": 0.0, "acc_1.25^2": 0.0, "acc_1.25^3": 0.0}
    comb_metrics = {"rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0, "acc_1.25": 0.0, "acc_1.25^2": 0.0, "acc_1.25^3": 0.0}
    sample_count = 0
    vis_count = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch is None:
                continue

            # Monocular evaluation: pass the same image as both inputs.
            input_img = batch["thermal"].to(device)
            rgb_img = batch["rgb"].to(device) if "rgb" in batch else None

            ft_outputs = ft_model(input_img, input_img)
            ft_pm1 = ft_outputs["pointmap1"]
            ft_pred_depth = ft_pm1[:, 2:3, :, :]

            base_outputs = base_model(input_img, input_img)
            base_pm1 = base_outputs["pointmap1"]
            base_pred_depth = base_pm1[:, 2:3, :, :]

            from dust3r.inference import inference
            dummy_instance = torch.zeros((1, 1, input_img.shape[-2], input_img.shape[-1]), device=device)
            image_pair = [(
                {"img": input_img, "instance": dummy_instance},
                {"img": input_img, "instance": dummy_instance}
            )]
            ma_out = inference(image_pair, ma_model, device, batch_size=1, verbose=False)
            if "pred1" not in ma_out:
                print("MASt3R inference failed for sample", i)
                continue
            ma_pred_pm1 = get_pointmap(ma_out["pred1"]).to(device)
            if ma_pred_pm1.dim() == 4 and ma_pred_pm1.shape[1] != 3:
                ma_pred_pm1 = ma_pred_pm1.permute(0, 3, 1, 2)
            ma_pred_depth = ma_pred_pm1[:, 2:3, :, :]

            combined_pred_depth = 0.5 * (base_pred_depth + ma_pred_depth)

            # Scale alignment
            if "gt_depth" in batch:
                gt_depth = batch["gt_depth"].to(device)
                valid_mask_ft = (gt_depth > 0) & (ft_pred_depth > 0) & torch.isfinite(gt_depth) & torch.isfinite(ft_pred_depth)
                scale_ft = (gt_depth[valid_mask_ft] * ft_pred_depth[valid_mask_ft]).sum() / ((ft_pred_depth[valid_mask_ft] ** 2).sum() + 1e-8) if valid_mask_ft.sum() > 100 else ref_scale

                valid_mask_comb = (gt_depth > 0) & (combined_pred_depth > 0) & torch.isfinite(gt_depth) & torch.isfinite(combined_pred_depth)
                scale_comb = (gt_depth[valid_mask_comb] * combined_pred_depth[valid_mask_comb]).sum() / ((combined_pred_depth[valid_mask_comb] ** 2).sum() + 1e-8) if valid_mask_comb.sum() > 100 else ref_scale
            else:
                scale_ft = scale_comb = ref_scale

            ft_depth_aligned = ft_pred_depth * scale_ft
            comb_depth_aligned = combined_pred_depth * scale_comb

            ft_depth_np = ft_depth_aligned[0].squeeze().cpu().numpy()
            comb_depth_np = comb_depth_aligned[0].squeeze().cpu().numpy()

            img_path = batch["thermal_path"][0]
            calib = load_calibrations(args.calib_yaml)
            intrinsics = get_intrinsics_from_yaml(calib, img_path)
            fx, fy, cx, cy = intrinsics
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            if "gt_depth" in batch:
                gt_depth = batch["gt_depth"].to(device)
                metrics_ft = compute_depth_metrics(ft_depth_aligned, gt_depth)
                metrics_comb = compute_depth_metrics(comb_depth_aligned, gt_depth)
                for k in ft_metrics.keys():
                    ft_metrics[k] += metrics_ft[k]
                    comb_metrics[k] += metrics_comb[k]
                sample_count += 1

            # Qualitative visualization
            if vis_count < args.vis_samples:
                if args.output_dir is None:
                    out_path = None
                    out_path_pcd = None
                else:
                    out_path = os.path.join(args.output_dir, f"vis_{i:04d}.png")
                    out_path_pcd = os.path.join(args.output_dir, f"comparison_pointcloud_{i:04d}.png")

                visualize_evaluation(
                    thermal_tensor=input_img[0],
                    rgb_tensor=rgb_img[0] if rgb_img is not None else None,
                    ft_depth=ft_depth_aligned[0],
                    comb_depth=comb_depth_aligned[0],
                    scale_ft=scale_ft,
                    scale_comb=scale_comb,
                    out_path=out_path,
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
                visualize_pointcloud(ft_depth_np, comb_depth_np, intrinsics, out_path=out_path_pcd, sample_rate=5)
                vis_count += 1

    if sample_count > 0:
        for k in ft_metrics:
            ft_metrics[k] /= sample_count
            comb_metrics[k] /= sample_count
        print("\n=== Evaluation Metrics (Freiburg Test) ===")
        print("Fine-tuned DUSt3R:")
        for k, v in ft_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("Combined (Base DUSt3R + MASt3R):")
        for k, v in comb_metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.output_dir is not None:
            metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write("Fine-tuned DUSt3R:\n")
                for k, v in ft_metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
                f.write("\nCombined (Base DUSt3R + MASt3R):\n")
                for k, v in comb_metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            print(f"Metrics saved to {metrics_path}")
    else:
        print("No ground-truth depth found or sample_count == 0. Skipping quantitative metrics.")

    print("Evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)