import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.gridspec import GridSpec
import cv2
import torch

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

def visualize_evaluation(thermal_tensor, rgb_tensor, ft_depth, comb_depth,
                         scale_ft=10.0, scale_comb=10.0, out_path="output.png",
                         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Visualize a 2x2 figure:
      Top-left: Input Thermal Image (enhanced for contrast)
      Top-right: Reference RGB (if available)
      Bottom-left: Fine-tuned predicted depth (color)
      Bottom-right: Combined (Base+MASt3R) predicted depth (color)
    Saves the figure to out_path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Process Thermal Image ---
    thermal_vis = thermal_tensor.permute(1, 2, 0).cpu().numpy()
    # Denormalize using std and mean:
    thermal_vis = thermal_vis * np.array(std) + np.array(mean)
    thermal_vis = np.clip(thermal_vis, 0, 1)
    # For a better thermal visualization, extract one channel and perform contrast stretch:
    thermal_gray = thermal_vis[..., 0]  # assuming all channels are similar
    valid_mask = thermal_gray > 0
    if valid_mask.sum() > 0:
        tmin, tmax = np.percentile(thermal_gray[valid_mask], [5, 95])
    else:
        tmin, tmax = thermal_gray.min(), thermal_gray.max()
    thermal_enhanced = np.clip((thermal_gray - tmin) / (tmax - tmin + 1e-8), 0, 1)

    # --- Process RGB if available ---
    rgb_vis = None
    if rgb_tensor is not None:
        rgb_vis = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_vis = rgb_vis * np.array(std) + np.array(mean)
        rgb_vis = np.clip(rgb_vis, 0, 1)

    # --- Process Depth Maps ---
    # Fine-tuned depth:
    ft_depth_np = ft_depth.squeeze().cpu().numpy()
    ft_depth_np = np.nan_to_num(ft_depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    valid_ft = ft_depth_np > 0
    if valid_ft.sum() > 0:
        dmin_ft, dmax_ft = np.percentile(ft_depth_np[valid_ft], [5, 95])
    else:
        dmin_ft, dmax_ft = ft_depth_np.min(), ft_depth_np.max()
    ft_norm = np.clip((ft_depth_np - dmin_ft) / (dmax_ft - dmin_ft + 1e-8), 0, 1)

    # Combined depth:
    comb_depth_np = comb_depth.squeeze().cpu().numpy()
    comb_depth_np = np.nan_to_num(comb_depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    valid_comb = comb_depth_np > 0
    if valid_comb.sum() > 0:
        dmin_comb, dmax_comb = np.percentile(comb_depth_np[valid_comb], [5, 95])
    else:
        dmin_comb, dmax_comb = comb_depth_np.min(), comb_depth_np.max()
    comb_norm = np.clip((comb_depth_np - dmin_comb) / (dmax_comb - dmin_comb + 1e-8), 0, 1)

    # --- Create 2x2 Figure ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Top-left: Enhanced Thermal image (displayed with 'inferno' colormap)
    axs[0, 0].imshow(thermal_enhanced, cmap='inferno')
    axs[0, 0].set_title("Input Thermal")
    axs[0, 0].axis("off")

    # Top-right: Reference RGB
    if rgb_vis is not None:
        axs[0, 1].imshow(rgb_vis)
        axs[0, 1].set_title("Reference RGB")
    else:
        axs[0, 1].text(0.5, 0.5, "No RGB", ha='center', va='center')
    axs[0, 1].axis("off")

    # Bottom-left: Fine-tuned Depth (Color)
    im1 = axs[1, 0].imshow(ft_norm, cmap="viridis")
    axs[1, 0].set_title(f"Fine-tuned Depth\n(scale: {scale_ft:.2f})")
    axs[1, 0].axis("off")
    fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Bottom-right: Combined Depth (Color)
    im2 = axs[1, 1].imshow(comb_norm, cmap="viridis")
    axs[1, 1].set_title(f"Base DUST3R + MAST3R Depth\n(scale: {scale_comb:.2f})")
    axs[1, 1].axis("off")
    fig.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
        print(f"Saved visualization: {out_path}")
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud(depth_map1, depth_map2, intrinsics, out_path="comparison_pointcloud.png", sample_rate=5):
    """
    Converts two depth maps to 3D point clouds using camera intrinsics and visualizes them side-by-side.
    """
    H, W = depth_map1.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Create mesh grid
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    # Downsample
    uu_ds = uu[::sample_rate, ::sample_rate]
    vv_ds = vv[::sample_rate, ::sample_rate]
    depth1_ds = depth_map1[::sample_rate, ::sample_rate]
    depth2_ds = depth_map2[::sample_rate, ::sample_rate]
    
    # Convert first depth map
    X1 = (uu_ds - cx) / fx * depth1_ds
    Y1 = (vv_ds - cy) / fy * depth1_ds
    Z1 = depth1_ds
    X1, Y1, Z1 = X1.flatten(), Y1.flatten(), Z1.flatten()
    
    # Convert second depth map
    X2 = (uu_ds - cx) / fx * depth2_ds
    Y2 = (vv_ds - cy) / fy * depth2_ds
    Z2 = depth2_ds
    X2, Y2, Z2 = X2.flatten(), Y2.flatten(), Z2.flatten()
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Left subplot: first point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(X1, Y1, Z1, c=Z1, cmap='viridis', s=1)
    ax1.set_title("Fine-tuned DUSt3R Point Cloud")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Depth (m)")
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.6, label="Depth (m)")
    ax1.view_init(elev=20, azim=-60)
    
    # Right subplot: second point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(X2, Y2, Z2, c=Z2, cmap='viridis', s=1)
    ax2.set_title("Combined (Base DUSt3R + MASt3R) Point Cloud")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Depth (m)")
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.6, label="Depth (m)")
    ax2.view_init(elev=20, azim=-60)
    
    # Adjust layout
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved combined point cloud visualization: {out_path}")
    plt.close(fig)