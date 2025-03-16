import os
import glob
import pickle
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

class FreiburgTrainThermalDataset(Dataset):
    def __init__(self, root_dir, annotations_path, transform=None, img_size=(224, 224), show_progress=False, cache_file="dataset_cache.pkl"):
        """
        Dataset for training DUSt3R with thermal images and pseudo annotations.
        
        Args:
            root_dir (str): Path to the dataset root directory.
            annotations_path (str): Path to the directory containing pseudo annotations (.npy files).
            transform (callable, optional): Optional transform to be applied to images.
            img_size (tuple): Target image size (width, height) for resizing.
            show_progress (bool): Whether to print progress messages.
            cache_file (str): Path to the cache file for storing sample metadata.
        """
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.img_size = img_size
        self.show_progress = show_progress
        self.transform = transform
        self.cache_file = cache_file
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        # Check if cache exists
        if os.path.exists(self.cache_file):
            if self.show_progress:
                print(f"Loading dataset cache from {self.cache_file} ...")
            with open(self.cache_file, "rb") as f:
                self.samples = pickle.load(f)
            if self.show_progress:
                print(f"Loaded {len(self.samples)} samples from cache.")
            return

        # Find all annotation files
        annotation_files = sorted(glob.glob(os.path.join(self.annotations_path, "*.npy")))
        
        # Wrap with tqdm if progress display is enabled
        if self.show_progress:
            from tqdm import tqdm
            annotation_files = tqdm(annotation_files, desc="Loading thermal dataset")
        
        for ann_path in annotation_files:
            try:
                # Load annotation to get frame paths
                annotation = np.load(ann_path, allow_pickle=True).item()
                
                if 'frame1_path' not in annotation:
                    print(f"Warning: annotation {ann_path} missing frame1_path")
                    continue
                
                # Get the RGB frame path from annotation
                rgb_frame_path = annotation['frame1_path']
                
                # Convert RGB path to thermal path by replacing 'fl_rgb' with 'fl_ir_aligned'
                thermal_path1 = rgb_frame_path.replace('fl_rgb', 'fl_ir_aligned')
                
                # For thermal_path2, we need to find the next thermal image in sequence
                thermal_dir = os.path.dirname(thermal_path1)
                thermal_files = sorted(glob.glob(os.path.join(thermal_dir, "*.png")))
                
                try:
                    current_index = thermal_files.index(thermal_path1)
                    if current_index + 1 < len(thermal_files):
                        thermal_path2 = thermal_files[current_index + 1]
                        
                        # Verify both files exist
                        if os.path.exists(thermal_path1) and os.path.exists(thermal_path2):
                            self.samples.append({
                                'thermal_path1': thermal_path1,
                                'thermal_path2': thermal_path2,
                                'annotation_path': ann_path
                            })
                    else:
                        if self.show_progress:
                            print(f"No second thermal image for {thermal_path1}")
                except ValueError:
                    print(f"Warning: Couldn't find {thermal_path1} in directory listing")
                    continue
            
            except Exception as e:
                print(f"Error processing annotation {ann_path}: {e}")
                continue

        print(f"FreiburgTrainThermalDataset: Found {len(self.samples)} valid thermal image pairs with annotations")
        
        # Save the samples list to cache
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.samples, f)
        if self.show_progress:
            print(f"Saved dataset cache to {self.cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            thermal_path1 = sample['thermal_path1']
            thermal_path2 = sample['thermal_path2']
            annotation_path = sample['annotation_path']

            # Load annotation
            annotation = np.load(annotation_path, allow_pickle=True).item()
            
            # Load thermal images
            thermal_img1 = self._load_thermal_image(thermal_path1)
            thermal_img2 = self._load_thermal_image(thermal_path2)
            
            # Extract directly stored pointmaps from annotation
            pointmap1 = annotation['pointmap1']  # Expected shape: [3, H, W]
            pointmap2 = annotation['pointmap2']  # Expected shape: [3, H, W]
            
            # Extract additional data from annotation
            depth_value_1 = annotation['depth_value_1']  # Z component of pointmap1
            intrinsics = annotation['intrinsics']
            pose1 = annotation['pose1']
            pose2 = annotation['pose2']
            
            # Convert pointmaps to tensors
            gt_pointmap1 = torch.from_numpy(pointmap1).float()
            gt_pointmap2 = torch.from_numpy(pointmap2).float()
            
            # Resize pointmaps if the image size differs
            if self.img_size:
                h, w = thermal_img1.shape[1], thermal_img1.shape[2]
                if gt_pointmap1.shape[1] != h or gt_pointmap1.shape[2] != w:
                    resized_pointmap1 = torch.zeros((3, h, w), dtype=torch.float32)
                    resized_pointmap2 = torch.zeros((3, h, w), dtype=torch.float32)
                    
                    for c in range(3):
                        channel1 = gt_pointmap1[c].numpy()
                        channel2 = gt_pointmap2[c].numpy()
                        resized_channel1 = cv2.resize(channel1, (w, h), interpolation=cv2.INTER_LINEAR)
                        resized_channel2 = cv2.resize(channel2, (w, h), interpolation=cv2.INTER_LINEAR)
                        resized_pointmap1[c] = torch.from_numpy(resized_channel1)
                        resized_pointmap2[c] = torch.from_numpy(resized_channel2)
                    
                    gt_pointmap1 = resized_pointmap1
                    gt_pointmap2 = resized_pointmap2
                    depth_value_1 = cv2.resize(depth_value_1, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Apply transform to thermal images if provided
            if self.transform:
                thermal_img1 = self.transform(thermal_img1)
                thermal_img2 = self.transform(thermal_img2)
            
            return {
                'img1': thermal_img1,  # Tensor shape: [3, H, W]
                'img2': thermal_img2,  # Tensor shape: [3, H, W]
                'gt_pointmap1': gt_pointmap1,  # Tensor shape: [3, H, W]
                'gt_pointmap2': gt_pointmap2,  # Tensor shape: [3, H, W]
                'gt_depth': torch.from_numpy(depth_value_1).float().unsqueeze(0),  # Tensor shape: [1, H, W]
                'intrinsics': torch.from_numpy(intrinsics).float(),  # Tensor shape: [3, 3]
                'pose1': torch.from_numpy(pose1).float(),  # Tensor shape: [4, 4]
                'pose2': torch.from_numpy(pose2).float(),  # Tensor shape: [4, 4]
                'thermal_path1': thermal_path1,
                'thermal_path2': thermal_path2
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            raise e
    
    def _load_thermal_image(self, path):
        """
        Load thermal image and convert to 3-channel tensor.
        """
        thermal_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if thermal_img is None:
            raise RuntimeError(f"Failed to load thermal image: {path}")
        thermal_img = thermal_img.astype(np.float32)
        thermal_img /= 65535.0  # Normalize assuming 16-bit image.
        thermal_img_3ch = np.stack([thermal_img, thermal_img, thermal_img], axis=-1)
        if self.img_size is not None:
            thermal_img_3ch = cv2.resize(thermal_img_3ch, self.img_size)
        thermal_tensor = torch.from_numpy(thermal_img_3ch).permute(2, 0, 1).float()
        thermal_tensor = pad_to_multiple_tensor(thermal_tensor, multiple=16)
        return thermal_tensor