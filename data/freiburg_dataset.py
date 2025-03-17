import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

def pad_to_multiple_tensor(img, multiple=16):
    """
    Pads a 3D tensor (C, H, W) so that its H and W become multiples of `multiple`.
    Returns the padded tensor.
    """
    _, H, W = img.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple
    pad_bottom = new_H - H
    pad_right = new_W - W
    # For a tensor of shape [C, H, W], F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padded = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom))
    return padded

class FreiburgThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=None):
        """
        Expects structure:
          seq_XX_day/00/fl_ir_aligned/*.png
                        /fl_rgb/*.png
        
        img_size: (width, height) tuple; if provided, images are resized accordingly.
        transform: Optional transform function to apply on images.
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        sequences = sorted(os.listdir(self.root_dir))
        for seq in sequences:
            seq_path = os.path.join(self.root_dir, seq)
            if not os.path.isdir(seq_path):
                continue
            
            drive_folders = sorted(
                d for d in os.listdir(seq_path)
                if os.path.isdir(os.path.join(seq_path, d))
            )
            
            for drive in drive_folders:
                drive_path = os.path.join(seq_path, drive)
                ir_dir = os.path.join(drive_path, 'fl_ir_aligned')
                rgb_dir = os.path.join(drive_path, 'fl_rgb')
                
                if not os.path.isdir(ir_dir) or not os.path.isdir(rgb_dir):
                    continue
                
                # Build a dict from the IR filenames (unique_part -> full_path)
                ir_dict = {}
                for f in glob.glob(os.path.join(ir_dir, '*.png')):
                    base = os.path.basename(f)  # e.g. "fl_ir_aligned_1570722156_952177040.png"
                    # remove the prefix "fl_ir_aligned_" so we get "1570722156_952177040.png"
                    unique_part = base.replace('fl_ir_aligned_', '')
                    ir_dict[unique_part] = f
                
                # For each RGB file, find a matching IR file by substring
                for f in glob.glob(os.path.join(rgb_dir, '*.png')):
                    base = os.path.basename(f)  # e.g. "fl_rgb_1570722156_952177040.png"
                    # remove prefix "fl_rgb_" so we get "1570722156_952177040.png"
                    unique_part = base.replace('fl_rgb_', '')
                    
                    if unique_part in ir_dict:
                        self.samples.append({
                            'thermal_path': ir_dict[unique_part],
                            'rgb_path': f,
                            'seq_name': seq,
                            'drive': drive
                        })
        
        print(f"FreiburgThermalDataset: Found {len(self.samples)} IR/RGB pairs in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:

            sample = self.samples[idx]
            thermal_path = sample['thermal_path']
            rgb_path = sample['rgb_path']

            if not os.path.exists(rgb_path):
                print(f"RGB file does not exist: {rgb_path}")
                return None
            if not os.path.exists(thermal_path):
                print(f"Thermal file does not exist: {thermal_path}")
                return None

            # Load IR image (can be 16-bit or 8-bit) and convert to float
            thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
            if thermal_img is None:
                raise RuntimeError(f"Failed to load thermal image: {thermal_path}")
            thermal_img = thermal_img.astype(np.float32)
            # Normalize (assuming 16-bit image)
            thermal_img /= 65535.0  
            # Convert to 3 channels by replication
            thermal_img_3ch = np.stack([thermal_img, thermal_img, thermal_img], axis=-1)

            # Load RGB image (BGR to RGB conversion)
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_img is None:
                raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # Optional resizing if img_size is provided (img_size is (width, height))
            if self.img_size is not None:
                thermal_img_3ch = cv2.resize(thermal_img_3ch, self.img_size)
                rgb_img = cv2.resize(rgb_img, self.img_size)

            # Apply transform if provided, else convert to torch tensor
            if self.transform:
                thermal_img_3ch = self.transform(thermal_img_3ch)
                rgb_img = self.transform(rgb_img)
            else:
                thermal_img_3ch = torch.from_numpy(thermal_img_3ch).permute(2, 0, 1).float()
                rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float()

            # Pad the images so that their height and width are multiples of 16
            thermal_img_3ch = pad_to_multiple_tensor(thermal_img_3ch, multiple=16)
            rgb_img = pad_to_multiple_tensor(rgb_img, multiple=16)

            return {
                'thermal': thermal_img_3ch,  # 3-channel IR image
                'thermal_path': thermal_path,
                'rgb': rgb_img,              # RGB image
                'rgb_path': rgb_path,
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None