# utils/camera_utils.py
import numpy as np
import torch
import cv2

def compute_pose_from_pointmaps(pointmap1, pointmap2):
    """
    Compute relative camera pose from pointmaps using Umeyama alignment
    
    Args:
        pointmap1: Pointmap from first image (C x H x W)
        pointmap2: Pointmap from second image (C x H x W)
        
    Returns:
        4x4 transformation matrix
    """
    # Reshape pointmaps to Nx3
    points1 = pointmap1.reshape(3, -1).transpose()
    points2 = pointmap2.reshape(3, -1).transpose()
    
    # Filter valid points (non-zero)
    valid_mask = (np.abs(points1).sum(axis=1) > 0) & (np.abs(points2).sum(axis=1) > 0)
    if valid_mask.sum() < 10:
        # Not enough points, return identity
        return np.eye(4)
    
    points1 = points1[valid_mask]
    points2 = points2[valid_mask]
    
    # Use Umeyama algorithm (SVD-based)
    T = umeyama_alignment(points1, points2)
    return T

def umeyama_alignment(points1, points2):
    """
    Computes the transformation that aligns points2 to points1
    using Umeyama's method (SVD-based)
    
    Args:
        points1: Nx3 array of points
        points2: Nx3 array of points
        
    Returns:
        4x4 transformation matrix
    """
    # Center the points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    
    # Compute cross-covariance matrix
    H = centered2.T @ centered1
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = U @ Vt
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    
    # Compute translation
    t = centroid1 - R @ centroid2
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def depth_to_pointmap(depth, intrinsics):
    """
    Convert depth map to 3D point map
    
    Args:
        depth: HxW depth map
        intrinsics: 3x3 camera intrinsics matrix
        
    Returns:
        3xHxW point map
    """
    h, w = depth.shape
    
    # Create pixel coordinates grid
    y, x = np.mgrid[:h, :w]
    
    # Extract intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Convert to normalized coordinates
    x_normalized = (x - cx) / fx
    y_normalized = (y - cy) / fy
    
    # Create point map
    point_map = np.zeros((3, h, w))
    point_map[0, :, :] = x_normalized * depth  # X
    point_map[1, :, :] = y_normalized * depth  # Y
    point_map[2, :, :] = depth  # Z
    
    return point_map