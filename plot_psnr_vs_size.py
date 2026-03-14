import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings


# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": "Times New Roman"
# })
# Configuration
eval_path = "./eval_mip360_indoor"
scenes = ["bonsai", "counter", "room", "kitchen"]
iteration = "30000"

# Color mapping for different methods
method_colors = {
    "beta": "#e60808",  # blue
    "nasg_gabor": "#890040",  # magenta
    "nasg": "#c11a68",  # green
    "asg": "#cd598f",   # purple
    "fb6": "#cd81a4",   # orange
    "slog": "#d7a9bf",  # brown
    "sh":  "#8b6678"    # blue
}

# Method display names
method_names = {
    "beta": "Spherical Beta",
    "nasg_gabor": "NASG Gabor",
    "nasg": "NASG",
    "asg": "ASG",
    "fb6": "FB6",
    "slog": "SLOG",
    "sh": "SH"
}


def get_file_size_megabytes(file_path):
    """Get the file size in megabytes (MB)."""
    try:
        file_size_bytes = os.path.getsize(file_path)
        # Convert bytes to megabytes: bytes / 1,000,000
        megabytes = file_size_bytes / 1_000_000
        return megabytes
    except Exception as e:
        warnings.warn(f"Error reading file size {file_path}: {e}")
        return None


def parse_directory_name(dir_name):
    """Parse directory name to extract method, degree, and scene."""
    # Pattern: method_[deg#_]scene
    # Examples: beta_kitchen, fb6_deg1_bonsai, nasg_gabor_deg2_room
    
    # Try patterns with degree
    pattern_with_deg = r"^(.+?)_deg(\d+)_(.+)$"
    match = re.match(pattern_with_deg, dir_name)
    if match:
        method = match.group(1)
        if method not in method_colors:
            return None, None, None
        degree = int(match.group(2))
        scene = match.group(3)
        return method, degree, scene
    
    # Try pattern without degree (e.g., beta_kitchen)
    pattern_no_deg = r"^(.+?)_(.+)$"
    match = re.match(pattern_no_deg, dir_name)
    if match:
        method = match.group(1)
        scene = match.group(2)
        # Beta doesn't have degrees, set to None or 0
        return method, 2, scene
    
    return None, None, None


def collect_data(eval_path, scenes):
    """Collect all data from evaluation results."""
    data = {scene: [] for scene in scenes}
    
    eval_dir = Path(eval_path)
    if not eval_dir.exists():
        warnings.warn(f"Evaluation directory not found: {eval_path}")
        return data
    
    # Iterate through all subdirectories
    for subdir in eval_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        dir_name = subdir.name
        method, degree, scene = parse_directory_name(dir_name)
        
        if scene not in scenes:
            continue
        
        if method is None:
            continue
        
        # Construct paths
        metrics_path = subdir / "point_cloud" / f"iteration_{iteration}" / "metrics.json"
        ply_path = subdir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        
        # Read metrics
        if not metrics_path.exists():
            warnings.warn(f"Metrics file not found: {metrics_path}")
            continue
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                psnr = metrics.get("PSNR")
        except Exception as e:
            warnings.warn(f"Error reading metrics from {metrics_path}: {e}")
            continue
        
        # Read point cloud file size
        if not ply_path.exists():
            warnings.warn(f"PLY file not found: {ply_path}")
            continue
        
        file_size_mb = get_file_size_megabytes(ply_path)
        if file_size_mb is None:
            warnings.warn(f"Could not read file size from {ply_path}")
            continue
        
        # Store data
        data[scene].append({
            "method": method,
            "degree": degree,
            "psnr": psnr,
            "size": file_size_mb,
            "label": f"d{degree}" if degree is not None else ""
        })
    
    return data


def plot_scene(scene_data, scene_name, output_path):
    """Create a plot for a single scene."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group data by method
    method_data = {}
    for point in scene_data:
        method = point["method"]
        if method not in method_data:
            method_data[method] = []
        method_data[method].append(point)
    
    # Plot each method
    for method, points in method_data.items():
        if method not in method_colors:
            continue
        
        # for degree 0, add also horizontal line for better visualization
        if method == "nasg" and points[0]["degree"] == 0:
            ax.axhline(points[0]["psnr"], color="#4b2135", linestyle='--', alpha=0.5)
        
        # Sort by size for better visualization
        points = sorted(points, key=lambda x: x["size"])
        
        sizes = [p["size"] for p in points]
        psnrs = [p["psnr"] for p in points]
        labels = [p["label"] for p in points]

        if method == "nasg" and points[0]["degree"] == 0:
            ax.plot(sizes[0], psnrs[0], 'o', color="#4b2135", label="d0", markersize=8)
            # Add label for horizontal line
            ax.annotate("d0", (sizes[0], psnrs[0]), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=9, 
                        color="#4b2135", weight='bold')
            continue
        
        # Plot line and markers
        ax.plot(sizes, psnrs, 'o-', color=method_colors[method], 
                label=method_names.get(method, method), linewidth=2, markersize=8)
        
        
        # Add labels to points
        for size, psnr, label in zip(sizes, psnrs, labels):
            if label:  # Only add label if not empty
                ax.annotate(label, (size, psnr), textcoords="offset points", 
                           xytext=(0, 8), ha='center', fontsize=9, 
                           color=method_colors[method], weight='bold')
    
    ax.set_xlabel("File Size (MB)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    if scene_name.lower() != "average (indoor scenes)":
        ax.set_title(f"PSNR vs File Size - {scene_name.capitalize()}", fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    order = ["beta", "nasg_gabor", "nasg", "asg", "fb6", "slog", "sh"]
    order_lables = ['Spherical Beta', 'NASG Gabor', 'NASG', 'ASG', 'FB6', 'SLOG', 'SH']
    ax.legend(handles=[handles[labels.index(method_names[m])] for m,l in zip(order, order_lables) if l in labels],
              labels=[method_names[m] for m, l in zip(order, order_lables) if l in labels], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")


def compute_average_data(data, scenes):
    """Compute average PSNR across all scenes for each method/degree combination."""
    # Group by method and degree
    avg_data = {}
    
    for scene in scenes:
        for point in data[scene]:
            key = (point["method"], point["degree"])
            if key not in avg_data:
                avg_data[key] = {"psnrs": [], "sizes": [], "method": point["method"], "degree": point["degree"]}
            avg_data[key]["psnrs"].append(point["psnr"])
            avg_data[key]["sizes"].append(point["size"])
    
    # Compute averages
    avg_results = []
    for key, values in avg_data.items():
        avg_results.append({
            "method": values["method"],
            "degree": values["degree"],
            "psnr": np.mean(values["psnrs"]),
            "size": np.mean(values["sizes"]),
            "label": f"d{values['degree']}" if values["degree"] is not None else ""
        })
    
    return avg_results


def main():
    print("=" * 80)
    print("PSNR vs File Size Analysis")
    print("=" * 80)
    
    # Collect data
    print("\nCollecting data from evaluation results...")
    data = collect_data(eval_path, scenes)
    
    # Create output directory for plots
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    
    # Plot individual scenes
    print("\nGenerating plots for individual scenes...")
    for scene in scenes:
        if not data[scene]:
            warnings.warn(f"No data found for scene: {scene}")
            continue
        
        output_path = output_dir / f"psnr_vs_size_{scene}.png"
        plot_scene(data[scene], scene, output_path)
    
    # Compute and plot average
    print("\nGenerating average plot...")
    avg_data = compute_average_data(data, scenes)
    if avg_data:
        output_path = output_dir / "psnr_vs_size_average.png"
        plot_scene(avg_data, "Average (Indoor Scenes)", output_path)
    else:
        warnings.warn("No data available for average plot")
    
    print("\n" + "=" * 80)
    print(f"✓ All plots saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
