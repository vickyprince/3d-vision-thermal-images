from torch.utils.data.dataloader import default_collate
import numpy as np

def get_pointmap(pred):
    for key in ["pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"]:
        if key in pred:
            return pred[key]
    raise KeyError("No recognized pointmap key found. Keys: " + str(list(pred.keys())))

def umeyama_alignment(points1, points2):
    """
    Computes the 4x4 rigid transformation that aligns points2 onto points1
    using Umeyama's method (SVD-based), ignoring any global scale factor.
    """
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2

    # Cross-covariance matrix
    H = centered2.T @ centered1

    # SVD
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Handle potential reflection
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Translation
    t = centroid1 - R @ centroid2

    # Build final 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_relative_pose_from_pointmaps(pointmap1, pointmap2):
    """
    Given two 3D pointmaps (shape [3, H, W]), compute a 4x4 transformation
    that maps points in 'pointmap2' into the coordinate system of 'pointmap1'
    via Umeyama alignment.
    """
    # Reshape to Nx3
    pts1 = pointmap1.reshape(3, -1).T  # shape (N, 3)
    pts2 = pointmap2.reshape(3, -1).T  # shape (N, 3)

    # Filter out invalid or zero points
    valid_mask = (np.abs(pts1).sum(axis=1) > 0) & (np.abs(pts2).sum(axis=1) > 0)
    pts1 = pts1[valid_mask]
    pts2 = pts2[valid_mask]

    # If not enough points, return identity
    if pts1.shape[0] < 10:
        return np.eye(4)

    # Use Umeyama
    T = umeyama_alignment(pts1, pts2)
    return T


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)