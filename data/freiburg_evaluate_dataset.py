import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def pad_to_multiple_tensor(img, multiple=16):
    """
    Pad a 3D tensor (C, H, W) so that its height and width become multiples of `multiple`.

    Args:
        img (torch.Tensor): Input tensor with shape [C, H, W].
        multiple (int): Desired multiple for height and width.

    Returns:
        torch.Tensor: Padded tensor.
    """
    _, H, W = img.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple
    pad_bottom = new_H - H
    pad_right = new_W - W
    return torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom))

class FreiburgEvaluateThermalDataset(Dataset):
    """
    Dataset for evaluating monocular thermal depth estimation on the Freiburg dataset.

    Expected directory structure:
        test/
          day/
            ImagesIR/   *.png
            ImagesRGB/  *.png    (optional, as reference)
            Depth/      *.png    (optional, if available)
          night/
            ImagesIR/   *.png
            ImagesRGB/  *.png
            Depth/      *.png

    This version loads each IR image (and corresponding RGB image, if available) individually.
    """
    def __init__(self, root_dir, transform=None, img_size=(224, 224)):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transformation to apply to images.
            img_size (tuple, optional): Desired image size as (width, height).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """
        Collect sample file paths for IR, RGB, and (optionally) depth images.
        """
        for condition in ["day", "night"]:
            condition_path = os.path.join(self.root_dir, condition)
            if not os.path.isdir(condition_path):
                continue

            ir_dir = os.path.join(condition_path, "ImagesIR")
            rgb_dir = os.path.join(condition_path, "ImagesRGB")
            depth_dir = os.path.join(condition_path, "Depth")  # Optional depth directory
            has_depth = os.path.isdir(depth_dir)

            if not os.path.isdir(ir_dir):
                print(f"Warning: {condition_path} missing 'ImagesIR'")
                continue

            ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.png")))
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) if os.path.isdir(rgb_dir) else []
            depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png"))) if has_depth else []

            for i, ir_file in enumerate(ir_files):
                sample = {"thermal_path": ir_file, "condition": condition}
                if i < len(rgb_files):
                    sample["rgb_path"] = rgb_files[i]
                if i < len(depth_files):
                    sample["depth_path"] = depth_files[i]
                self.samples.append(sample)

        print(f"FreiburgEvaluateThermalDataset: Found {len(self.samples)} samples in {self.root_dir}")
        print(f"Has depth GT: {any('depth_path' in s for s in self.samples)}")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'thermal': 3-channel thermal image as a torch.Tensor.
                - 'thermal_path': Path to the thermal image.
                - 'rgb': RGB image as a torch.Tensor (if available).
                - 'rgb_path': Path to the RGB image (if available).
                - 'gt_depth': Ground-truth depth as a torch.Tensor (if available).
                - 'depth_path': Path to the depth image (if available).
            Returns None if an error occurs.
        """
        sample = self.samples[idx]
        ir_path = sample["thermal_path"]

        # Load and normalize the IR image.
        ir_img = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
        if ir_img is None:
            raise RuntimeError(f"Failed to load IR image: {ir_path}")
        ir_img = ir_img.astype(np.float32) / 65535.0
        ir_img_3ch = np.stack([ir_img, ir_img, ir_img], axis=-1)
        if self.img_size is not None:
            w, h = self.img_size
            ir_img_3ch = cv2.resize(ir_img_3ch, (w, h))
        ir_tensor = torch.from_numpy(ir_img_3ch).permute(2, 0, 1)
        ir_tensor = pad_to_multiple_tensor(ir_tensor, 16)
        if self.transform:
            ir_tensor = self.transform(ir_tensor)

        result = {"thermal": ir_tensor, "thermal_path": ir_path}

        # Load RGB image if available.
        if "rgb_path" in sample:
            rgb_path = sample["rgb_path"]
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_img is not None:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if self.img_size is not None:
                    rgb_img = cv2.resize(rgb_img, (w, h))
                rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1)
                rgb_tensor = pad_to_multiple_tensor(rgb_tensor, 16)
                if self.transform:
                    rgb_tensor = self.transform(rgb_tensor)
                result["rgb"] = rgb_tensor
                result["rgb_path"] = rgb_path

        # Load depth image if available.
        if "depth_path" in sample:
            depth_path = sample["depth_path"]
            try:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is not None:
                    depth_img = depth_img.astype(np.float32) / 1000.0
                    if self.img_size is not None:
                        depth_img = cv2.resize(depth_img, (w, h))
                    depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)
                    depth_tensor = pad_to_multiple_tensor(depth_tensor, 16)
                    result["gt_depth"] = depth_tensor
                    result["depth_path"] = depth_path
            except Exception as e:
                print(f"Error loading depth {depth_path}: {e}")

        # Fallback: if RGB is not available, set it to the thermal image.
        if "rgb" not in result:
            result["rgb"] = ir_tensor

        return result