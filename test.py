import os
import numpy as np
from utils.visualization import visualize_annotation_correctness

# Specify the path to your annotations folder
annotations_dir = "data/annotations"
# Optionally, you can specify a directory where you want to save the visualizations
viz_save_dir = "data/annotation_visualizations"
os.makedirs(viz_save_dir, exist_ok=True)

# Get a list of annotation files (npy files)
annotation_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith(".npy")]

# Let's visualize a few samples (e.g., first 5)
for ann_file in annotation_files[:5]:
    # Load the annotation dictionary
    annotation = np.load(ann_file, allow_pickle=True).item()
    
    # Use the 'frame1_path' from the annotation as the basis for the RGB image.
    # This is based on your annotation-generation script.
    rgb_path = annotation.get("frame1_path")
    
    # Derive the thermal path (assuming a naming convention where 'fl_rgb' is replaced by 'fl_ir_aligned')
    if rgb_path is not None:
        thermal_path = rgb_path.replace("fl_rgb", "fl_ir_aligned")
    else:
        # If not available, set thermal_path to None (or handle appropriately)
        thermal_path = None
    
    # Define a save path for the visualization image
    base_name = os.path.splitext(os.path.basename(ann_file))[0]
    save_path = os.path.join(viz_save_dir, f"{base_name}_viz.png")
    
    # Call the visualization function
    visualize_annotation_correctness(annotation, rgb_path, thermal_path, save_path=save_path)
    print(f"Saved visualization for {ann_file} at {save_path}")