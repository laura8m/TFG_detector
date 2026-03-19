#!/usr/bin/env python3
"""
Test standalone de range_projection.py sin ROS 2.

Extrae la lógica de procesamiento para comparar con lidar_pipeline_suite.py.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import patchwork++
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/build/patchworkpp')
import pypatchworkpp

def load_kitti_scan(bin_path):
    """Load KITTI .bin file."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]  # xyz only

def load_semantickitti_labels(label_path):
    """Load SemanticKITTI labels."""
    if not Path(label_path).exists():
        return None
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF
    return semantic_labels

def project_to_range_image_simple(points, H=64, W=2048, fov_up=3.0, fov_down=-25.0):
    """
    Simple range image projection (without full Node logic).

    Returns:
        range_image: (H, W) array with ranges
        u, v: (N,) pixel coordinates
        valid_mask: (N,) boolean mask
    """
    N = len(points)

    # Calculate range
    r = np.sqrt(np.sum(points**2, axis=1))

    # Angles
    pitch = np.arcsin(np.clip(points[:, 2] / r, -1.0, 1.0))
    yaw = np.arctan2(points[:, 1], points[:, 0])

    # Convert to pixels
    fov_up_rad = fov_up * np.pi / 180.0
    fov_down_rad = fov_down * np.pi / 180.0
    fov_total = fov_up_rad - fov_down_rad

    # Row (u)
    proj_y = (pitch - fov_down_rad) / fov_total
    proj_y = 1.0 - proj_y
    u = np.floor(proj_y * H).astype(np.int32)
    u = np.clip(u, 0, H - 1)

    # Column (v)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    v = np.floor(proj_x * W).astype(np.int32)
    v = np.clip(v, 0, W - 1)

    # Valid mask
    valid_mask = (r > 1.0) & (r < 80.0)

    # Create range image (closest point wins)
    range_image = np.zeros((H, W), dtype=np.float32)

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        order = np.argsort(r[valid_idx])[::-1]  # Descending order

        u_sorted = u[valid_idx][order]
        v_sorted = v[valid_idx][order]
        r_sorted = r[valid_idx][order]

        range_image[u_sorted, v_sorted] = r_sorted

    return range_image, u, v, valid_mask, r

def compute_delta_r_simple(points, patchwork, threshold_obs=-0.3):
    """
    Compute delta_r using Patchwork++ (simplified, no wall rejection).

    Returns:
        delta_r: (N,) delta_r values
        obs_mask: (N,) boolean mask for obstacles
    """
    N = len(points)

    # Run Patchwork++
    patchwork.estimateGround(points)
    ground_indices = patchwork.getGroundIndices()

    # Get local planes (placeholder - would need full CZM implementation)
    # For now, use simple ground plane
    ground_pts = points[ground_indices]

    if len(ground_pts) < 3:
        # No ground found, assume all obstacles
        return np.full(N, -1.0), np.ones(N, dtype=bool)

    # Fit plane to ground points
    centroid = np.mean(ground_pts, axis=0)
    _, _, vh = np.linalg.svd(ground_pts - centroid)
    normal = vh[2, :]

    # Ensure normal points up
    if normal[2] < 0:
        normal = -normal

    d = -np.dot(normal, centroid)

    # Calculate delta_r for all points
    r = np.sqrt(np.sum(points**2, axis=1))
    ray_dir = points / r[:, np.newaxis]
    dot_prod = np.dot(ray_dir, normal)

    # Avoid division by zero
    valid_dot = np.abs(dot_prod) > 1e-6
    r_expected = np.full(N, np.inf)
    r_expected[valid_dot] = -d / dot_prod[valid_dot]

    delta_r = r - r_expected

    # Obstacle mask
    obs_mask = delta_r < threshold_obs

    return delta_r, obs_mask

def test_range_projection_standalone(scan_idx=0):
    """
    Test range_projection.py logic standalone.
    """
    print("="*80)
    print("TEST: range_projection.py STANDALONE")
    print("="*80)

    # Paths
    data_root = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/00/00")
    velodyne_dir = data_root / "velodyne"
    labels_dir = data_root / "labels"

    scan_path = velodyne_dir / f"{scan_idx:06d}.bin"
    label_path = labels_dir / f"{scan_idx:06d}.label"

    if not scan_path.exists():
        print(f"[ERROR] Scan not found: {scan_path}")
        return

    # Load data
    print(f"\n✓ Loading scan {scan_idx}: {scan_path}")
    points = load_kitti_scan(scan_path)
    print(f"  Total points: {len(points)}")

    # Load ground truth
    gt_labels = load_semantickitti_labels(label_path)
    if gt_labels is not None:
        # SemanticKITTI obstacle classes: 252 (moving-car), 10-19 (vehicles), 30-39 (person, bicyclist)
        obstacle_classes = [10, 11, 13, 15, 18, 20, 30, 31, 32, 252]
        gt_obstacles = np.isin(gt_labels, obstacle_classes)
        n_gt_obstacles = np.sum(gt_obstacles)
        print(f"  Ground truth obstacles: {n_gt_obstacles}")
    else:
        gt_obstacles = None
        print(f"  Ground truth labels not found")

    # Initialize Patchwork++
    params = pypatchworkpp.Parameters()
    params.verbose = False
    patchwork = pypatchworkpp.patchworkpp(params)

    # Stage 1: Ground segmentation + delta_r
    print(f"\n--- Stage 1: Ground Segmentation + Delta-r ---")
    t0 = time.time()
    delta_r, obs_mask_raw = compute_delta_r_simple(points, patchwork)
    t1 = time.time()

    n_obs_raw = np.sum(obs_mask_raw)
    print(f"  Obstacles detected (delta_r < -0.3): {n_obs_raw}")
    print(f"  Timing: {(t1-t0)*1000:.1f} ms")

    # Project to range image
    print(f"\n--- Stage 2: Range Image Projection ---")
    t0 = time.time()
    range_image, u, v, valid_mask, r = project_to_range_image_simple(points)
    t1 = time.time()

    n_valid_pixels = np.sum(range_image > 0)
    n_valid_points = np.sum(valid_mask)
    compression_ratio = n_valid_points / n_valid_pixels if n_valid_pixels > 0 else 0

    print(f"  Valid points: {n_valid_points}")
    print(f"  Unique pixels: {n_valid_pixels}")
    print(f"  Compression ratio: {compression_ratio:.1f}:1")
    print(f"  Timing: {(t1-t0)*1000:.1f} ms")

    # Project delta_r to range image (like range_projection.py)
    print(f"\n--- Stage 3: Delta-r → Range Image (Binary Probability) ---")
    delta_r_image = np.zeros((64, 2048), dtype=np.float32)

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        order = np.argsort(r[valid_idx])[::-1]

        u_sorted = u[valid_idx][order]
        v_sorted = v[valid_idx][order]
        delta_r_sorted = delta_r[valid_idx][order]

        delta_r_image[u_sorted, v_sorted] = delta_r_sorted

    # Binary probability (as in range_projection.py line 1000)
    P_raw_binary = (delta_r_image < -0.3).astype(np.float32)
    n_obs_pixels = np.sum(P_raw_binary > 0.5)

    print(f"  Obstacle pixels (delta_r < -0.3): {n_obs_pixels} / {n_valid_pixels}")

    # Map back to points
    obs_mask_from_image = P_raw_binary[u[valid_mask], v[valid_mask]] > 0.5
    obs_points_from_image = np.zeros(len(points), dtype=bool)
    obs_points_from_image[valid_mask] = obs_mask_from_image
    n_obs_from_image = np.sum(obs_points_from_image)

    print(f"  Obstacles as points: {n_obs_from_image}")

    # Evaluate against ground truth
    if gt_obstacles is not None:
        print(f"\n--- Evaluation ---")

        # Per-point evaluation
        tp = np.sum(obs_mask_raw & gt_obstacles)
        fp = np.sum(obs_mask_raw & ~gt_obstacles)
        fn = np.sum(~obs_mask_raw & gt_obstacles)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  Per-Point Metrics (Stage 1: delta_r):")
        print(f"    Precision: {precision*100:.2f}%")
        print(f"    Recall:    {recall*100:.2f}%")
        print(f"    F1 Score:  {f1*100:.2f}%")
        print(f"    TP: {tp}, FP: {fp}, FN: {fn}")

        # Range image evaluation
        tp_img = np.sum(obs_points_from_image & gt_obstacles)
        fp_img = np.sum(obs_points_from_image & ~gt_obstacles)
        fn_img = np.sum(~obs_points_from_image & gt_obstacles)

        precision_img = tp_img / (tp_img + fp_img) if (tp_img + fp_img) > 0 else 0
        recall_img = tp_img / (tp_img + fn_img) if (tp_img + fn_img) > 0 else 0
        f1_img = 2 * precision_img * recall_img / (precision_img + recall_img) if (precision_img + recall_img) > 0 else 0

        print(f"\n  Range Image Metrics (Stage 3: binary probability):")
        print(f"    Precision: {precision_img*100:.2f}%")
        print(f"    Recall:    {recall_img*100:.2f}%")
        print(f"    F1 Score:  {f1_img*100:.2f}%")
        print(f"    TP: {tp_img}, FP: {fp_img}, FN: {fn_img}")

        # Compare loss
        recall_loss = (recall - recall_img) * 100
        print(f"\n  Recall Loss (per-point → range image): {recall_loss:.2f}%")

    print("\n" + "="*80)
    print("TEST COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=int, default=0, help='Scan index')
    args = parser.parse_args()

    test_range_projection_standalone(scan_idx=args.scan)
