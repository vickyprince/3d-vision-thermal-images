import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.gridspec import GridSpec
import cv2

def visualize_annotation_correctness(annotation, rgb_path, thermal_path, save_path=None, sample_points=5000):
    """
    Visualize the complete pseudo-annotation to verify correctness:
      - Left (reduced width): 3D pointcloud with drawn camera axes (pose1 and pose2)
      - Top-Right: RGB image
      - Bottom-Right: Depth map (Z channel from pointmap)
    """
    # --- Load RGB image ---
    try:
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading RGB image: {rgb_path}, error={e}")
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # --- Load and process Thermal image (unused but kept for potential debugging) ---
    try:
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
        if thermal_img is None:
            raise ValueError("Thermal image is None")
        t_min, t_max = thermal_img.min(), thermal_img.max()
        thermal_float = (thermal_img.astype(np.float32) - t_min) / (t_max - t_min + 1e-8)
        thermal_8u = (thermal_float * 255).astype(np.uint8)
        thermal_eq = cv2.equalizeHist(thermal_8u)
    except Exception as e:
        print(f"Error processing thermal image: {thermal_path}, error={e}")
        thermal_eq = np.zeros((100, 100), dtype=np.uint8)
    
    # --- Retrieve annotation data ---
    # Expected keys:
    #   'pointmap1'     : [3, H, W] or [H, W, 3]
    #   'depth_value_1' : [H, W]
    #   'pose1'         : 4x4 identity (canonical)
    #   'pose2'         : 4x4 relative pose
    pm1 = annotation['pointmap1']
    depth_value = annotation['depth_value_1']
    pose1 = annotation['pose1']
    pose2 = annotation['pose2']
    
    # --- Create 3D pointcloud from pointmap1 ---
    # If pm1 is channels-last ([H, W, 3]), then transpose to channels-first.
    if pm1.shape[0] != 3 and pm1.shape[-1] == 3:
        pm1 = pm1.transpose(2, 0, 1)
    points = pm1.reshape(3, -1).T  # shape (N, 3)
    
    # Randomly sample points if too large
    N = points.shape[0]
    if N > sample_points:
        idx = np.random.choice(N, sample_points, replace=False)
        points_sampled = points[idx]
    else:
        points_sampled = points
    
    # --- Helper: Draw camera axes in 3D ---
    def draw_camera_axes(ax, T, scale=0.1, label=''):
        origin = T[:3, 3]
        x_axis = T[:3, :3] @ np.array([scale, 0, 0])
        y_axis = T[:3, :3] @ np.array([0, scale, 0])
        z_axis = T[:3, :3] @ np.array([0, 0, scale])
        ax.quiver(origin[0], origin[1], origin[2],
                  x_axis[0], x_axis[1], x_axis[2],
                  color='r', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  y_axis[0], y_axis[1], y_axis[2],
                  color='g', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  z_axis[0], z_axis[1], z_axis[2],
                  color='b', linewidth=2)
        ax.text(origin[0], origin[1], origin[2], label, color='k')
    
    # --- Set up figure with GridSpec ---
    # Updated layout: 2 rows x 2 columns, with left column smaller and right column larger.
    # For example, set width_ratios to [1, 1.5]
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1.5])
    
    # Left: 3D pointcloud (spanning both rows)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    sc = ax3d.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                      s=1, c=points_sampled[:, 2], cmap='viridis')
    ax3d.set_title("3D Pointcloud with Camera Axes")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    draw_camera_axes(ax3d, pose1, scale=0.1, label='Cam1')
    draw_camera_axes(ax3d, pose2, scale=0.1, label='Cam2')
    fig.colorbar(sc, ax=ax3d, fraction=0.02, pad=0.08)
    
    # Top-Right: RGB image
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")
    
    # Bottom-Right: Depth map (Z channel)
    ax_depth = fig.add_subplot(gs[1, 1])
    im_depth = ax_depth.imshow(depth_value, cmap='plasma', vmin=0, vmax=20)
    ax_depth.set_title("Depth Map (Z from Pointmap)")
    ax_depth.axis("off")
    fig.colorbar(im_depth, ax=ax_depth, fraction=0.02, pad=0.08)
    
    plt.suptitle("Annotation Visualization & Verification", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()