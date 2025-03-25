import os
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

def pad_to_multiple_tensor(img, multiple=16):
    """
    Pad a 3D tensor (C, H, W) so that its height and width become multiples of the given value.

    Args:
        img (torch.Tensor): Input tensor of shape [C, H, W].
        multiple (int): The multiple to pad the height and width to.

    Returns:
        torch.Tensor: The padded tensor.
    """
    _, H, W = img.shape
    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple
    pad_bottom = new_H - H
    pad_right = new_W - W
    padded = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom))
    return padded

class FreiburgTrainThermalDataset(Dataset):
    """
    Dataset for training DUSt3R with thermal images and pseudo annotations.
    
    Expected directory structure:
        seq_XX_day/00/
            fl_ir_aligned/   *.png
            fl_rgb/          *.png

    Args:
        root_dir (str): Root directory of the dataset.
        annotations_path (str): Directory containing pseudo annotations (.npy or .npz files).
        transform (callable, optional): Transformation to apply to images.
        augmentation (callable, optional): Augmentation to apply to images.
        img_size (tuple): Target image size (width, height) for resizing.
        show_progress (bool): If True, prints progress messages.
        cache_file (str): Path to cache file to store sample metadata.
    """
    def __init__(self, root_dir, annotations_path, transform=None, augmentation=None,
                 img_size=(224, 224), show_progress=False, cache_file="dataset_cache.pkl"):
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.img_size = img_size
        self.show_progress = show_progress
        self.transform = transform
        self.augmentation = augmentation
        self.cache_file = cache_file
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """
        Collect sample file paths by matching pseudo annotation files with corresponding
        thermal image pairs.
        """
        if os.path.exists(self.cache_file):
            if self.show_progress:
                print(f"Loading dataset cache from {self.cache_file} ...")
            with open(self.cache_file, "rb") as f:
                self.samples = pickle.load(f)
            if self.show_progress:
                print(f"Loaded {len(self.samples)} samples from cache.")
            return

        annotation_files = sorted(glob.glob(os.path.join(self.annotations_path, "*.np*")))
        if self.show_progress:
            from tqdm import tqdm
            annotation_files = tqdm(annotation_files, desc="Loading thermal dataset")
        
        for ann_path in annotation_files:
            try:
                annotation = np.load(ann_path, allow_pickle=True).item()
                if 'frame1_path' not in annotation:
                    print(f"Warning: annotation {ann_path} missing frame1_path")
                    continue
                rgb_frame_path = annotation['frame1_path']
                thermal_path1 = rgb_frame_path.replace('fl_rgb', 'fl_ir_aligned')
                thermal_dir = os.path.dirname(thermal_path1)
                thermal_files = sorted(glob.glob(os.path.join(thermal_dir, "*.png")))
                try:
                    current_index = thermal_files.index(thermal_path1)
                    if current_index + 1 < len(thermal_files):
                        thermal_path2 = thermal_files[current_index + 1]
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
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.samples, f)
        if self.show_progress:
            print(f"Saved dataset cache to {self.cache_file}")

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Returns:
            dict: A dictionary containing:
                - 'img1': Thermal image tensor from first view.
                - 'img2': Thermal image tensor from second view.
                - 'gt_pointmap1': Ground-truth pointmap from first view.
                - 'gt_pointmap2': Ground-truth pointmap from second view.
                - 'gt_depth': Ground-truth depth map (as 1xHxW tensor).
                - 'intrinsics': Camera intrinsics matrix as a tensor.
                - 'pose1': Camera pose for the first view.
                - 'pose2': Relative pose from first to second view.
                - 'thermal_path1': Path to first thermal image.
                - 'thermal_path2': Path to second thermal image.
            Returns None if an error occurs.
        """
        try:
            sample = self.samples[idx]
            thermal_path1 = sample['thermal_path1']
            thermal_path2 = sample['thermal_path2']
            annotation_path = sample['annotation_path']

            annotation = np.load(annotation_path, allow_pickle=True).item()

            thermal_img1 = self._load_thermal_image(thermal_path1)
            thermal_img2 = self._load_thermal_image(thermal_path2)
            
            if self.augmentation:
                thermal_img1 = self.augmentation(thermal_img1)
                thermal_img2 = self.augmentation(thermal_img2)
            
            if self.transform:
                thermal_img1 = self.transform(thermal_img1)
                thermal_img2 = self.transform(thermal_img2)
            
            pointmap1 = annotation['pointmap1']
            pointmap2 = annotation['pointmap2']
            if pointmap1.shape[-1] == 3 and pointmap1.shape[0] != 3:
                pointmap1 = pointmap1.transpose(2, 0, 1)
            if pointmap2.shape[-1] == 3 and pointmap2.shape[0] != 3:
                pointmap2 = pointmap2.transpose(2, 0, 1)

            gt_pointmap1 = torch.from_numpy(pointmap1).float()
            gt_pointmap2 = torch.from_numpy(pointmap2).float()
            depth_value_1 = annotation['depth_value_1']
            intrinsics = annotation['intrinsics']
            pose1 = annotation['pose1']
            pose2 = annotation['pose2']

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

                    if depth_value_1 is None or not isinstance(depth_value_1, np.ndarray):
                        print(f"Warning: Invalid depth map in annotation {annotation_path}, using zeros.")
                        depth_value_1 = np.zeros((h, w), dtype=np.float32)
                    elif depth_value_1.size == 0 or depth_value_1.ndim != 2:
                        print(f"Warning: Empty or invalid shape depth map in annotation {annotation_path}, using zeros.")
                        depth_value_1 = np.zeros((h, w), dtype=np.float32)
                    else:
                        try:
                            depth_value_1 = depth_value_1.astype(np.float32)
                            depth_value_1 = cv2.resize(depth_value_1, (w, h), interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            print(f"Warning: cv2.resize failed for depth map in annotation {annotation_path}: {e}")
                            depth_value_1 = np.zeros((h, w), dtype=np.float32)
            
            return {
                'img1': thermal_img1,
                'img2': thermal_img2,
                'gt_pointmap1': gt_pointmap1,
                'gt_pointmap2': gt_pointmap2,
                'gt_depth': torch.from_numpy(depth_value_1).float().unsqueeze(0),
                'intrinsics': torch.from_numpy(intrinsics).float(),
                'pose1': torch.from_numpy(pose1).float(),
                'pose2': torch.from_numpy(pose2).float(),
                'thermal_path1': thermal_path1,
                'thermal_path2': thermal_path2
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    def _load_thermal_image(self, path):
        """
        Load a thermal image, normalize, resize, and pad it.
        
        Args:
            path (str): Path to the thermal image.
        
        Returns:
            torch.Tensor: Processed thermal image tensor.
        """
        thermal_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if thermal_img is None:
            raise RuntimeError(f"Failed to load thermal image: {path}")
        thermal_img = thermal_img.astype(np.float32)
        thermal_img /= 65535.0
        thermal_img_3ch = np.stack([thermal_img, thermal_img, thermal_img], axis=-1)
        if self.img_size is not None:
            thermal_img_3ch = cv2.resize(thermal_img_3ch, self.img_size)
        thermal_tensor = torch.from_numpy(thermal_img_3ch).permute(2, 0, 1).float()
        thermal_tensor = pad_to_multiple_tensor(thermal_tensor, multiple=16)
        return thermal_tensor