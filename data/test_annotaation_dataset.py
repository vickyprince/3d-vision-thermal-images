import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

def pad_to_multiple_tensor(img, multiple=16):
    """
    Pad a 3D tensor (C, H, W) so that H and W become multiples of 'multiple'.

    Args:
        img (torch.Tensor): Input tensor with shape [C, H, W].
        multiple (int): The value for which H and W should be multiples.

    Returns:
        torch.Tensor: Padded tensor.
    """
    _, H, W = img.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple
    pad_bottom = new_H - H
    pad_right = new_W - W
    padded = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom))
    return padded

class FreiburgThermalTestDataset(Dataset):
    """
    Dataset for loading test paired images from a test dataset.
    
    Expected directory structure:
        root_dir/
            day/
                ImagesIR/*.png
                ImagesRGB/*.png        (optional)
                Depth/*.png            (optional)
            night/
                ImagesIR/*.png
                ImagesRGB/*.png        (optional)
                Depth/*.png            (optional)
    """
    def __init__(self, root_dir, transform=None, img_size=None):
        """
        Args:
            root_dir (str): Root directory of the test dataset.
            transform (callable, optional): Transformation to apply to images.
            img_size (tuple, optional): Desired image size as (width, height).
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """
        Collect samples from both 'day' and 'night' directories.
        For each condition, IR images are mandatory.
        RGB and depth images are optional.
        Matching is done by the sorted order of files.
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

            # Match each IR file with the corresponding RGB/depth file (by index)
            for i, ir_file in enumerate(ir_files):
                sample = {'thermal_path': ir_file, 'condition': condition}
                sample['rgb_path'] = rgb_files[i] if i < len(rgb_files) else None
                if has_depth:
                    sample['depth_path'] = depth_files[i] if i < len(depth_files) else None
                self.samples.append(sample)

        print(f"FreiburgThermalTestDataset: Found {len(self.samples)} samples in {self.root_dir}")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a test sample by index.

        Returns:
            dict: Dictionary with keys:
                  - 'thermal': 3-channel thermal image as a torch.Tensor.
                  - 'thermal_path': Path to the thermal image.
                  - 'rgb': RGB image as a torch.Tensor (if available).
                  - 'rgb_path': Path to the RGB image (if available).
                  - 'depth': Depth image as a torch.Tensor (if available).
                  - 'depth_path': Path to the depth image (if available).
                  - 'condition': The condition ("day" or "night").
            Returns None if an error occurs.
        """
        sample = self.samples[idx]
        thermal_path = sample.get('thermal_path')
        rgb_path = sample.get('rgb_path')
        depth_path = sample.get('depth_path', None)

        # Load and normalize thermal image.
        if not os.path.exists(thermal_path):
            print(f"Thermal file does not exist: {thermal_path}")
            return None
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
        if thermal_img is None:
            raise RuntimeError(f"Failed to load thermal image: {thermal_path}")
        thermal_img = thermal_img.astype(np.float32) / 65535.0
        # Convert single channel to 3-channel image.
        thermal_img_3ch = np.stack([thermal_img, thermal_img, thermal_img], axis=-1)

        # Load RGB image if available.
        rgb_img = None
        if rgb_path is not None:
            if not os.path.exists(rgb_path):
                print(f"RGB file does not exist: {rgb_path}")
            else:
                rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if rgb_img is None:
                    raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load depth image if available.
        depth_img = None
        if depth_path is not None:
            if not os.path.exists(depth_path):
                print(f"Depth file does not exist: {depth_path}")
            else:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    raise RuntimeError(f"Failed to load depth image: {depth_path}")
                depth_img = depth_img.astype(np.float32)
                # Optionally expand dims if depth is single-channel.
                if len(depth_img.shape) == 2:
                    depth_img = np.expand_dims(depth_img, axis=-1)

        # Resize images if an image size is provided.
        if self.img_size is not None:
            thermal_img_3ch = cv2.resize(thermal_img_3ch, self.img_size)
            if rgb_img is not None:
                rgb_img = cv2.resize(rgb_img, self.img_size)
            if depth_img is not None:
                depth_img = cv2.resize(depth_img, self.img_size)

        # Apply transformation if provided; else convert images to torch.Tensor.
        if self.transform:
            thermal_img_3ch = self.transform(thermal_img_3ch)
            if rgb_img is not None:
                rgb_img = self.transform(rgb_img)
            if depth_img is not None:
                depth_img = self.transform(depth_img)
        else:
            thermal_img_3ch = torch.from_numpy(thermal_img_3ch).permute(2, 0, 1).float()
            if rgb_img is not None:
                rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
            if depth_img is not None:
                # If depth image is single channel, ensure it has a channel dimension.
                if len(depth_img.shape) == 2 or (len(depth_img.shape) == 3 and depth_img.shape[-1] == 1):
                    depth_img = torch.from_numpy(depth_img).unsqueeze(0).float()
                else:
                    depth_img = torch.from_numpy(depth_img).permute(2, 0, 1).float()

        # Pad images so that dimensions are multiples of 16.
        thermal_img_3ch = pad_to_multiple_tensor(thermal_img_3ch, multiple=16)
        if rgb_img is not None:
            rgb_img = pad_to_multiple_tensor(rgb_img, multiple=16)
        if depth_img is not None:
            depth_img = pad_to_multiple_tensor(depth_img, multiple=16)

        out = {
            'thermal': thermal_img_3ch,
            'thermal_path': thermal_path,
            'condition': sample.get('condition')
        }
        if rgb_img is not None:
            out['rgb'] = rgb_img
            out['rgb_path'] = rgb_path
        if depth_img is not None:
            out['depth'] = depth_img
            out['depth_path'] = depth_path

        return out