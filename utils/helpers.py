from torch.utils.data.dataloader import default_collate
import numpy as np
import yaml
import os

def get_pointmap(pred):
    """
    Extract the pointmap from a prediction dictionary.

    Searches for the keys:
      "pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"

    Args:
        pred (dict): Dictionary containing prediction data.

    Returns:
        The pointmap corresponding to one of the recognized keys.

    Raises:
        KeyError: If none of the recognized keys are found.
    """
    for key in ["pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"]:
        if key in pred:
            return pred[key]
    raise KeyError("No recognized pointmap key found. Keys: " + str(list(pred.keys())))

def umeyama_alignment(points1, points2):
    """
    Compute the 4x4 rigid transformation matrix that aligns points2 onto points1
    using Umeyama's method (SVD-based), ignoring any global scale factor.

    Args:
        points1 (np.ndarray): An (N, 3) array of points.
        points2 (np.ndarray): An (N, 3) array of points.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2

    # Cross-covariance matrix
    H = centered2.T @ centered1

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Correct reflection if necessary
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = centroid1 - R @ centroid2

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def compute_relative_pose_from_pointmaps(pointmap1, pointmap2):
    """
    Compute a 4x4 transformation matrix that maps points in 'pointmap2' 
    into the coordinate system of 'pointmap1' using Umeyama alignment.

    Args:
        pointmap1 (np.ndarray): Pointmap from the first image of shape (3, H, W).
        pointmap2 (np.ndarray): Pointmap from the second image of shape (3, H, W).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    pts1 = pointmap1.reshape(3, -1).T  # (N, 3)
    pts2 = pointmap2.reshape(3, -1).T  # (N, 3)

    # Filter valid points (non-zero)
    valid_mask = (np.abs(pts1).sum(axis=1) > 0) & (np.abs(pts2).sum(axis=1) > 0)
    pts1 = pts1[valid_mask]
    pts2 = pts2[valid_mask]

    if pts1.shape[0] < 10:
        return np.eye(4)

    T = umeyama_alignment(pts1, pts2)
    return T

def custom_collate(batch):
    """
    Custom collate function for DataLoader. Filters out any None items
    from the batch before applying the default collate function.

    Args:
        batch (list): List of dataset items.

    Returns:
        Collated batch.
    """
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

def get_intrinsics_from_yaml(calib, image_path):
    """
    Retrieve camera intrinsics from calibration data based on the image filename.

    If the filename contains 'fl_rgb', the left intrinsics are used; otherwise, the right intrinsics are returned.

    Args:
        calib (dict): Calibration data loaded from YAML.
        image_path (str): Path of the image file.

    Returns:
        np.ndarray: Camera intrinsics matrix.
    """
    base = os.path.basename(image_path).lower()
    if "fl_rgb" in base:
        intrinsics = np.array(calib["left"]["intrinsics"])
    else:
        intrinsics = np.array(calib["right"]["intrinsics"])
    return intrinsics

def load_calibrations(calib_yaml):
    """
    Load calibration data from a YAML file.

    Args:
        calib_yaml (str): Path to the calibration YAML file.

    Returns:
        dict: Calibration data.
    """
    with open(calib_yaml, "r") as f:
        calib = yaml.safe_load(f)
    return calib