import os
from argparse import ArgumentParser
import json
import pandas as pd
import torch
import time

# Scenes to evaluate
scenes = ["counter", "kitchen", "bonsai"]

# Cap max for each scene
cap_max = {
    "bonsai": 1_500_000,
    "room": 1_500_000,
    "counter": 1_500_000,
    "kitchen": 1_500_000,
}

outdoor_cap_max = {
    "bicycle": 6_000_000,
    "flowers": 3_000_000,
    "garden": 5_000_000,
    "stump": 4_500_000,
    "treehill": 3_500_000,
}

eval_iteration = "30_000"

parser = ArgumentParser(description="Mip360 Indoor Scenes Full Evaluation")
parser.add_argument(
    "--mipnerf360", "-m360", type=str, required=True, help="Path to Mip-NeRF360 dataset"
)
parser.add_argument("--output_path", default="./eval_mip360_indoor")
args = parser.parse_args()


# Helper function to create markdown tables and compute mean
def create_markdown_table(metrics, dataset_name):
    df = pd.DataFrame(metrics)
    cols = ["Scene", "PSNR", "SSIM", "LPIPS", "Training Time (s)"]
    df = df[cols]

    # Compute mean row
    mean_row = df.drop("Scene", axis=1).mean()
    mean_row["Scene"] = "Mean"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Generate markdown table
    md_table = (
        f"## Metrics for {dataset_name}\n"
        + df.to_markdown(index=False, floatfmt=".4f")
        + "\n\n"
    )
    return md_table

outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]

#Run spherical harmonics evaluation for 1 2 3 degrees 
print("\n" + "=" * 80)
print("SPHERICAL HARMONICS EVALUATION - DEGREES 1, 2, 3")
print("=" * 80)

sh_metrics = []
sh_training_times = {}
for degree in [3]:
    print(f"\n{'=' * 80}")
    print(f"Spherical Harmonics Degree {degree}")
    print(f"{'=' * 80}")
    
    for scene in outdoor_scenes:
        source = os.path.join(args.mipnerf360, scene)
        output = os.path.join(args.output_path, f"sh_deg{degree}_{scene}_outdoor")

        print(f"\nTraining {scene} with Spherical Harmonics (degree={degree})...")
        cmd = (
            f"python train.py -s {source} --images images_4 "
            f"-m {output} --cap_max {outdoor_cap_max[scene]} "
            f"--eval --disable_viewer --quiet "
            f"--rendering_mode beta --sb_number 0 --sh_degree {degree} --iterations {eval_iteration}"
        )
        print(f"Command: {cmd}")
        start_time = time.time()
        ret = os.system(cmd)
        end_time = time.time()
        training_time = end_time - start_time
        sh_training_times[scene] = training_time
        print(f"Training time: {training_time:.2f}s")
        if ret != 0:
            print(f"ERROR: Training failed for {scene} with Spherical Harmonics degree {degree}")
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# collect metrics for SH
for scene in outdoor_scenes:
    scene_path = os.path.join(args.output_path, f"sh_deg{degree}_{scene}_outdoor")
    results_file = os.path.join(
        scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
    )
    try:        
        with open(results_file, "r") as f:
            scene_metrics = json.load(f)
        scene_metrics["Scene"] = scene
        scene_metrics["Training Time (s)"] = sh_training_times.get(scene, 0)
        sh_metrics.append(scene_metrics)
    except FileNotFoundError:
        print(f"WARNING: Metrics file not found for {scene} (Spherical Harmonics degree {degree})")
# Save SH results
if sh_metrics:
    output_text = create_markdown_table(
        sh_metrics, f"Spherical Harmonics Degree {degree} (Mip360 Outdoor)"
    )
    output_file = os.path.join(args.output_path, f"sh_degree{degree}_metrics.txt")
    with open(output_file, "w") as f:
        f.write(output_text)
    print(f"\n✓ Spherical Harmonics Degree {degree} metrics saved to {output_file}")
    print(output_text)



# print("=" * 80)
# print("SPHERICAL BETA EVALUATION")
# print("=" * 80)

# # 1. Run Spherical Beta on all scenes
# beta_metrics = []
# beta_training_times = {}
# for scene in scenes:
#     source = os.path.join(args.mipnerf360, scene)
#     output = os.path.join(args.output_path, f"beta_{scene}")
    
#     print(f"\nTraining {scene} with Spherical Beta...")
#     cmd = (
#         f"python train.py -s {source} --images images_2 "
#         f"-m {output} --cap_max {cap_max[scene]} "
#         f"--eval --disable_viewer --quiet "
#         f"--rendering_mode beta --iterations {eval_iteration}"
#     )
#     print(f"Command: {cmd}")
#     start_time = time.time()
#     ret = os.system(cmd)
#     end_time = time.time()
#     training_time = end_time - start_time
#     beta_training_times[scene] = training_time
#     print(f"Training time: {training_time:.2f}s")
#     if ret != 0:
#         print(f"ERROR: Training failed for {scene} with beta")
    
#     # Clean up CUDA memory
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

# # Collect metrics for Spherical Beta
# for scene in scenes:
#     scene_path = os.path.join(args.output_path, f"beta_{scene}")
#     results_file = os.path.join(
#         scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#     )
#     try:
#         with open(results_file, "r") as f:
#             scene_metrics = json.load(f)
#         scene_metrics["Scene"] = scene
#         scene_metrics["Training Time (s)"] = beta_training_times.get(scene, 0)
#         beta_metrics.append(scene_metrics)
#     except FileNotFoundError:
#         print(f"WARNING: Metrics file not found for {scene} (beta)")

# # Save Spherical Beta results
# if beta_metrics:
#     output_text = create_markdown_table(beta_metrics, "Spherical Beta (Mip360 Indoor)")
#     output_file = os.path.join(args.output_path, "spherical_beta_metrics.txt")
#     with open(output_file, "w") as f:
#         f.write(output_text)
#     print(f"\n✓ Spherical Beta metrics saved to {output_file}")
#     print(output_text)

# print("\n" + "=" * 80)
# print("NASG (TORCH) EVALUATION - DEGREES 1, 2, 3")
# print("=" * 80)

# 2. Run NASG with degrees 1, 2, 3
# nasg_degrees = [0]
# for degree in nasg_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"NASG Degree {degree}")
#     print(f"{'=' * 80}")
    
#     nasg_metrics = []
#     nasg_training_times = {}
    
#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"nasg_deg{degree}_{scene}")
        
#         print(f"\nTraining {scene} with NASG (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer  "
#             f"--rendering_mode gmm --sb_number {degree} --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         nasg_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with NASG degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
    
#     # Collect metrics for this NASG degree
#     for scene in scenes:
#         scene_path = os.path.join(args.output_path, f"nasg_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = nasg_training_times.get(scene, 0)
#             nasg_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (NASG degree {degree})")
    
#     # Save NASG degree results
#     if nasg_metrics:
#         output_text = create_markdown_table(
#             nasg_metrics, f"NASG Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"nasg_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ NASG Degree {degree} metrics saved to {output_file}")
#         print(output_text)

# 3. Run NASG Gabor with degrees 1, 2, 3
# scenes = ["kitchen"]
# nasg_gabor_degrees = [1]
# for degree in nasg_gabor_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"NASG Gabor Degree {degree}")
#     print(f"{'=' * 80}")

#     nasg_gabor_metrics = []
#     nasg_gabor_training_times = {}

#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"nasg_gabor_deg{degree}_{scene}")

#         print(f"\nTraining {scene} with NASG Gabor (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode nasg_gabor --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         nasg_gabor_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with NASG Gabor degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     # Collect metrics for this NASG Gabor degree
#     for scene in scenes:    
#         scene_path = os.path.join(args.output_path, f"nasg_gabor_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = nasg_gabor_training_times.get(scene, 0)
#             nasg_gabor_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (NASG Gabor degree {degree})")
    
#     # Save NASG Gabor degree results
#     if nasg_gabor_metrics:
#         output_text = create_markdown_table(
#             nasg_gabor_metrics, f"NASG Gabor Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"nasg_gabor_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ NASG Gabor Degree {degree} metrics saved to {output_file}")
#         print(output_text)

# 4. Run FB6 with degrees 1, 2, 3
# fb6_degrees = [3]
# for degree in fb6_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"FB6 Degree {degree}")
#     print(f"{'=' * 80}")  # what does to fox say

#     fb6_metrics = []
#     fb6_training_times = {}

#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"fb6_deg{degree}_{scene}")

#         print(f"\nTraining {scene} with FB6 (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode fb6 --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         fb6_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with FB6 degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     # Collect metrics for this FB6 degree
#     for scene in scenes:    
#         scene_path = os.path.join(args.output_path, f"fb6_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"  # ring ding ding ding diing dingindignidnginsdign
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = fb6_training_times.get(scene, 0)
#             fb6_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (FB6 degree {degree})")
    
#     # Save FB6 degree results
#     if fb6_metrics:
#         output_text = create_markdown_table(
#             fb6_metrics, f"FB6 Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"fb6_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ FB6 Degree {degree} metrics saved to {output_file}")
#         print(output_text)

# # Scenes to evaluate
# scenes = ["bonsai"]

# asg_degrees = [2]
# for degree in asg_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"ASG Degree {degree}")
#     print(f"{'=' * 80}")

#     asg_metrics = []
#     asg_training_times = {}  # who let the dogs out

#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"asg_deg{degree}_{scene}")

#         print(f"\nTraining {scene} with ASG (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode asg --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         asg_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with ASG degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     # Collect metrics for this ASG degree
#     for scene in scenes:    
#         scene_path = os.path.join(args.output_path, f"asg_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = asg_training_times.get(scene, 0)
#             asg_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (ASG degree {degree})")

#     # Save ASG degree results
#     if asg_metrics:
#         output_text = create_markdown_table(
#             asg_metrics, f"ASG Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"asg_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ ASG Degree {degree} metrics saved to {output_file}")
#         print(output_text)




# scenes = ["bonsai", "room", "counter", "kitchen"]
# slog_degrees = [1, 2, 3]
# for degree in slog_degrees:
#     print(f"  - slog_degree{degree}_metrics.txt")
#     slog_metrics = []
#     slog_training_times = {}  # who let the dogs out

#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"slog_deg{degree}_{scene}")

#         print(f"\nTraining {scene} with SLOG (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode slog --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         slog_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with SLOG degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     # Collect metrics for this SLOG degree
#     for scene in scenes:    
#         scene_path = os.path.join(args.output_path, f"slog_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = slog_training_times.get(scene, 0)
#             slog_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (SLOG degree {degree})")

#     # Save SLOG degree results
#     if slog_metrics:
#         output_text = create_markdown_table(
#             slog_metrics, f"SLOG Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"slog_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ SLOG Degree {degree} metrics saved to {output_file}")
#         print(output_text)


    
# scenes = [ "room", "counter", "kitchen"]

# sg_degrees = [2, 3]
# for degree in sg_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"SG Degree {degree}")
#     print(f"{'=' * 80}")

#     sg_metrics = []
#     sg_training_times = {}  # who let the dogs out

#     for scene in scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"sg_deg{degree}_{scene}")

#         print(f"\nTraining {scene} with GS (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_2 "
#             f"-m {output} --cap_max {cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode sg --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         sg_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with SG degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     # Collect metrics for this SG degree
#     for scene in scenes:    
#         scene_path = os.path.join(args.output_path, f"sg_deg{degree}_{scene}")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = sg_training_times.get(scene, 0)
#             sg_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (SG degree {degree})")

#     # Save SG degree results
#     if sg_metrics:
#         output_text = create_markdown_table(
#             sg_metrics, f"SG Degree {degree} (Mip360 Indoor)"
#         )
#         output_file = os.path.join(args.output_path, f"sg_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ SG Degree {degree} metrics saved to {output_file}")
#         print(output_text)



# for deg in slog_degrees:
#     print(f"  - slog_degree{deg}_metrics.txt")

# ============================================================================
# OUTDOOR SCENES EVALUATION - Degree 1 only
# ============================================================================

# print("\n" + "=" * 80)
# print("OUTDOOR SCENES EVALUATION - DEGREE 1")
# print("=" * 80)

# # Outdoor scenes
# outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]

# # Cap max for outdoor scenes (typically larger than indoor)
# outdoor_cap_max = {
#     "bicycle": 6_000_000,
#     "flowers": 3_000_000,
#     "garden": 5_000_000,
#     "stump": 4_500_000,
#     "treehill": 3_500_000,
# }

# # Methods to evaluate on outdoor scenes (degree 1 only)
# outdoor_methods = [
#     # ("nasg", "nasg"),
#     ("nasg_gabor", "nasg_gabor"),
#     ("fb6", "fb6"),
#     ("asg", "asg"),
#     # ("sg", "sg"),
#     # ("slog", "slog"),
# ]

# nasg_degrees = [0]
# for degree in nasg_degrees:
#     print(f"\n{'=' * 80}")
#     print(f"NASG Degree {degree}")
#     print(f"{'=' * 80}")
    
#     nasg_metrics = []
#     nasg_training_times = {}
    
#     for scene in outdoor_scenes:
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"nasg_deg{degree}_{scene}_outdoor")
        
#         print(f"\nTraining {scene} with NASG (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_4 "
#             f"-m {output} --cap_max {outdoor_cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode nasg --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         nasg_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with NASG degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
    
#     # Collect metrics for this NASG degree
#     for scene in scenes:
#         scene_path = os.path.join(args.output_path, f"nasg_deg{degree}_{scene}_outdoor")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = nasg_training_times.get(scene, 0)
#             nasg_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} (NASG degree {degree})")
    
#     # Save NASG degree results
#     if nasg_metrics:
#         output_text = create_markdown_table(
#             nasg_metrics, f"NASG Degree {degree} (Mip360 Outdoor)"
#         )
#         output_file = os.path.join(args.output_path, f"nasg_degree{degree}_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ NASG Degree {degree} metrics saved to {output_file}")
#         print(output_text)

# degree = 1  # Fixed to degree 1 for outdoor scenes

# for method_name, gmm_mode in outdoor_methods:
#     print(f"\n{'=' * 80}")
#     print(f"{method_name.upper()} Degree {degree} - Outdoor Scenes")
#     print(f"{'=' * 80}")
    
#     method_metrics = []
#     method_training_times = {}
    
#     for scene in outdoor_scenes:
#         if method_name == "fb6":
#             scene = "treehill"
#         elif method_name == "asg":
#             scene = "stump"
#         source = os.path.join(args.mipnerf360, scene)
#         output = os.path.join(args.output_path, f"{method_name}_deg{degree}_{scene}_outdoor")
        
#         print(f"\nTraining {scene} with {method_name.upper()} (degree={degree})...")
#         cmd = (
#             f"python train.py -s {source} --images images_4 "
#             f"-m {output} --cap_max {outdoor_cap_max[scene]} "
#             f"--eval --disable_viewer --quiet "
#             f"--rendering_mode gmm --sb_number {degree} --gmm_color_mode {gmm_mode} --iterations {eval_iteration}"
#         )
#         print(f"Command: {cmd}")
#         start_time = time.time()
#         ret = os.system(cmd)
#         end_time = time.time()
#         training_time = end_time - start_time
#         method_training_times[scene] = training_time
#         print(f"Training time: {training_time:.2f}s")
#         if ret != 0:
#             print(f"ERROR: Training failed for {scene} with {method_name.upper()} degree {degree}")
        
#         # Clean up CUDA memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
    
#     # Collect metrics for this method
#     for scene in outdoor_scenes:    
#         scene_path = os.path.join(args.output_path, f"{method_name}_deg{degree}_{scene}_outdoor")
#         results_file = os.path.join(
#             scene_path, f"point_cloud/iteration_{eval_iteration}/metrics.json"
#         )
#         try:
#             with open(results_file, "r") as f:
#                 scene_metrics = json.load(f)
#             scene_metrics["Scene"] = scene
#             scene_metrics["Training Time (s)"] = method_training_times.get(scene, 0)
#             method_metrics.append(scene_metrics)
#         except FileNotFoundError:
#             print(f"WARNING: Metrics file not found for {scene} ({method_name.upper()} degree {degree})")
    
#     # Save method results
#     if method_metrics:
#         output_text = create_markdown_table(
#             method_metrics, f"{method_name.upper()} Degree {degree} (Mip360 Outdoor)"
#         )
#         output_file = os.path.join(args.output_path, f"{method_name}_degree{degree}_outdoor_metrics.txt")
#         with open(output_file, "w") as f:
#             f.write(output_text)
#         print(f"\n✓ {method_name.upper()} Degree {degree} outdoor metrics saved to {output_file}")
#         print(output_text)

# print("\n" + "=" * 80)
# print("OUTDOOR EVALUATION COMPLETE")
# print("=" * 80)