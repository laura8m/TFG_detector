#!/usr/bin/env python3
"""
Test debug para entender por qué wall rejection no funciona.
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


def load_kitti_bin(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]
    return points


def main():
    scan_id = 0
    data_path = f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne/{scan_id:06d}.bin"

    points = load_kitti_bin(data_path)
    print(f"Total points: {len(points)}\n")

    # Config con wall rejection
    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,
        verbose=True
    )

    print(f"Config:")
    print(f"  wall_rejection_slope: {config.wall_rejection_slope}")
    print(f"  wall_height_diff_threshold: {config.wall_height_diff_threshold}")
    print(f"  wall_kdtree_radius: {config.wall_kdtree_radius}\n")

    pipeline = LidarPipelineSuite(config, data_path=data_path)

    # Ejecutar segment_ground y analizar internals
    print("="*80)
    print("EJECUTANDO segment_ground() CON DEBUG")
    print("="*80 + "\n")

    ground_pts, n_per_point, d_per_point, rejected_mask = pipeline.segment_ground(points)

    print(f"\n--- OUTPUTS de segment_ground() ---")
    print(f"ground_pts shape: {ground_pts.shape}")
    print(f"rejected_mask shape: {rejected_mask.shape}")
    print(f"rejected_mask sum: {np.sum(rejected_mask)} (puntos marcados como wall)")
    print(f"rejected_mask dtype: {rejected_mask.dtype}")

    # Analizar local_planes y rejected_bins
    print(f"\n--- INTERNALS ---")
    print(f"local_planes: {len(pipeline.local_planes)} bins válidos")

    # Ahora ejecutar stage1_complete
    print(f"\n" + "="*80)
    print("EJECUTANDO stage1_complete()")
    print("="*80 + "\n")

    results = pipeline.stage1_complete(points)

    print(f"ground_indices: {len(results['ground_indices'])}")
    print(f"rejected_walls: {len(results['rejected_walls'])}")
    print(f"timing: {results['timing_ms']:.1f}ms")

    # Analizar por qué rejected_walls es 0
    ground_indices = pipeline.patchwork.getGroundIndices()
    print(f"\nDEBUG: ground_indices from Patchwork++: {len(ground_indices)}")
    print(f"DEBUG: rejected_mask[ground_indices] sum: {np.sum(rejected_mask[ground_indices])}")

    # Ver si algún centro fue rechazado
    centers = pipeline.patchwork.getCenters()
    normals = pipeline.patchwork.getNormals()

    print(f"\nDEBUG: Total centers: {len(centers)}")

    # Contar cuántos centros tienen nz < 0.7
    nz_values = normals[:, 2]
    suspect_normals = np.sum(np.abs(nz_values) < config.wall_rejection_slope)
    print(f"DEBUG: Centers con |nz| < {config.wall_rejection_slope}: {suspect_normals}")

    if suspect_normals > 0:
        # Analizar algunos
        suspect_idx = np.where(np.abs(nz_values) < config.wall_rejection_slope)[0]
        print(f"\nDEBUG: Primeros 5 centros sospechosos:")
        for i in suspect_idx[:5]:
            c = centers[i]
            n = normals[i]
            print(f"  Center {i}: pos={c}, nz={n[2]:.3f}")

if __name__ == '__main__':
    main()
