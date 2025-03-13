# utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def visualize_depth_map(depth_map, save_path=None, cmap='viridis', norm=True):
    """
    Visualize and optionally save a depth map
    
    Args:
        depth_map: Depth map array
        save_path: Path to save the visualization (optional)
        cmap: Colormap to use
        norm: Whether to normalize the depth map for visualization
    """
    # Copy depth map for visualization
    vis_depth = depth_map.copy()
    
    # Handle invalid values
    valid_mask = vis_depth > 0
    if valid_mask.sum() == 0:
        vis_depth = np.zeros_like(vis_depth)
    else:
        # Set invalid values to min valid value
        min_valid = vis_depth[valid_mask].min()
        vis_depth[~valid_mask] = min_valid
        
        # Normalize if requested
        if norm:
            vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min() + 1e-8)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_depth, cmap=cmap)
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        print("before ploting")
        plt.show()

def visualize_pointcloud(pointmap, save_path=None, color=None):
    """
    Visualize and optionally save a point cloud from a pointmap
    
    Args:
        pointmap: 3xHxW point map (X, Y, Z coordinates)
        save_path: Path to save the visualization (optional)
        color: Optional color for the point cloud
    """
    # Reshape pointmap to Nx3
    points = pointmap.reshape(3, -1).transpose()
    
    # Filter valid points (non-zero)
    valid_mask = np.abs(points).sum(axis=1) > 0
    points = points[valid_mask]
    
    if len(points) == 0:
        print("No valid points in the point cloud")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Set colors if provided
    if color is not None:
        if isinstance(color, np.ndarray) and color.shape[0] == len(points):
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            pcd.paint_uniform_color(color)
    else:
        # Color by Z coordinate (depth)
        z_values = points[:, 2]
        z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-8)
        
        # Create a color map from red (near) to blue (far)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = 1 - z_normalized  # Red (inversely proportional to depth)
        colors[:, 2] = z_normalized      # Blue (proportional to depth)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save if path provided
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
    else:
        # Visualize
        o3d.visualization.draw_geometries([pcd])