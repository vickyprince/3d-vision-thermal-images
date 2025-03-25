import numpy as np
import torch
import cv2

def compute_pose_from_pointmaps(pointmap1, pointmap2):
    """
    Compute the relative camera pose from two pointmaps using Umeyama alignment.
    
    Args:
        pointmap1 (np.ndarray): Pointmap from the first image of shape (C, H, W).
        pointmap2 (np.ndarray): Pointmap from the second image of shape (C, H, W).
        
    Returns:
        np.ndarray: 4x4 transformation matrix representing the relative pose.
    """
    # Reshape pointmaps to Nx3 arrays
    points1 = pointmap1.reshape(3, -1).transpose()
    points2 = pointmap2.reshape(3, -1).transpose()
    
    # Filter valid points (non-zero)
    valid_mask = (np.abs(points1).sum(axis=1) > 0) & (np.abs(points2).sum(axis=1) > 0)
    if valid_mask.sum() < 10:
        return np.eye(4)
    
    points1 = points1[valid_mask]
    points2 = points2[valid_mask]
    
    T = umeyama_alignment(points1, points2)
    return T

def umeyama_alignment(points1, points2):
    """
    Computes the transformation that aligns points2 to points1 using Umeyama's method.
    
    Args:
        points1 (np.ndarray): Nx3 array of points.
        points2 (np.ndarray): Nx3 array of points.
        
    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    
    H = centered2.T @ centered1
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    
    t = centroid1 - R @ centroid2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def depth_to_pointmap(depth, intrinsics):
    """
    Convert a depth map into a 3D point map using the camera intrinsics.
    
    Args:
        depth (np.ndarray): HxW depth map.
        intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        
    Returns:
        np.ndarray: 3xHxW point map where channels represent X, Y, and Z coordinates.
    """
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    x_normalized = (x - cx) / fx
    y_normalized = (y - cy) / fy
    
    point_map = np.zeros((3, h, w))
    point_map[0, :, :] = x_normalized * depth  # X coordinates
    point_map[1, :, :] = y_normalized * depth  # Y coordinates
    point_map[2, :, :] = depth                 # Z coordinates
    
    return point_map