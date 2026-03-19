#!/usr/bin/env python3
"""
Parameter sweep para wall rejection v2.1

Prueba diferentes combinaciones de:
  - delta_z_threshold: [0.15, 0.2, 0.25, 0.3]
  - kdtree_radius: [0.3, 0.5, 0.7]
  - min_neighbors: [3, 5, 8]
"""

import sys
import os
import numpy as np

# Añadir paths necesarios
patchwork_site_packages = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_site_packages)
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
os.chdir('/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')

import pypatchworkpp
from ring_anomaly_detection import estimate_local_ground_planes, _validate_and_reject_walls_pointwise


def load_kitti_bin(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    return scan.reshape((-1, 4))[:, :3]


def load_kitti_labels(file_path):
    labels = np.fromfile(file_path, dtype=np.uint32)
    return labels & 0xFFFF


def test_params(scan_id, delta_z, radius, min_neigh):
    """Test specific parameter combination"""
    # Cargar datos
    base_velodyne = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne"
    base_labels = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels"

    points = load_kitti_bin(f"{base_velodyne}/{scan_id:06d}.bin")
    labels = load_kitti_labels(f"{base_labels}/{scan_id:06d}.label")

    # Ground truth
    gt_wall_mask = np.isin(labels, [50, 51, 52])
    gt_wall_indices = np.where(gt_wall_mask)[0]

    gt_ground_mask = np.isin(labels, [40, 44, 48, 49, 72])
    gt_ground_indices = np.where(gt_ground_mask)[0]

    # Patchwork++
    pw_params = pypatchworkpp.Parameters()
    pw_params.verbose = False
    pw = pypatchworkpp.patchworkpp(pw_params)

    result = estimate_local_ground_planes(points, pw, patchwork_params=pw_params, enable_wall_rejection=False)
    ground_pw = result['ground_indices']

    # Targets: wall points que Patchwork++ clasificó mal
    fp_wall_mask = np.isin(ground_pw, gt_wall_indices)
    fp_wall_indices = ground_pw[fp_wall_mask]

    if len(fp_wall_indices) == 0:
        return None  # No hay nada que detectar

    # V2.1 con parámetros custom
    rejected = _validate_and_reject_walls_pointwise(
        points,
        ground_pw,
        delta_z_threshold=delta_z,
        use_percentiles=True,
        kdtree_radius=radius,
        min_neighbors=min_neigh
    )

    # Métricas
    tp = np.intersect1d(rejected, gt_wall_indices)
    fp = np.intersect1d(rejected, gt_ground_indices)

    recall = len(tp) / len(fp_wall_indices) * 100 if len(fp_wall_indices) > 0 else 0
    precision = len(tp) / len(rejected) * 100 if len(rejected) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'rejected': len(rejected),
        'tp': len(tp),
        'fp': len(fp),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'targets': len(fp_wall_indices)
    }


def parameter_sweep(scan_id=0):
    """Run full parameter sweep"""
    print("=" * 100)
    print(f"PARAMETER SWEEP - Wall Rejection v2.1 - Scan {scan_id:06d}")
    print("=" * 100)

    # Configuración de sweep
    delta_z_vals = [0.15, 0.2, 0.25, 0.3]
    radius_vals = [0.3, 0.5, 0.7]
    min_neigh_vals = [3, 5, 8]

    results = []

    print(f"\nTesting {len(delta_z_vals) * len(radius_vals) * len(min_neigh_vals)} configurations...")

    for delta_z in delta_z_vals:
        for radius in radius_vals:
            for min_neigh in min_neigh_vals:
                print(f"  δZ={delta_z:.2f}m, r={radius:.1f}m, min_n={min_neigh}...", end=" ")

                result = test_params(scan_id, delta_z, radius, min_neigh)

                if result is None:
                    print("SKIP (no targets)")
                    continue

                result.update({'delta_z': delta_z, 'radius': radius, 'min_neighbors': min_neigh})
                results.append(result)

                print(f"Recall={result['recall']:.1f}%, Prec={result['precision']:.1f}%, F1={result['f1']:.1f}")

    # Análisis de resultados
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGURATIONS (by F1-score)")
    print("=" * 100)

    # Ordenar por F1
    results.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n{'Rank':<5} {'δZ':<7} {'Radius':<8} {'Min_N':<7} {'Recall':<10} {'Prec':<10} {'F1':<10} "
          f"{'TP':<6} {'FP':<6} {'Total':<7}")
    print("-" * 100)

    for i, r in enumerate(results[:10], 1):
        print(f"{i:<5} {r['delta_z']:<7.2f} {r['radius']:<8.1f} {r['min_neighbors']:<7} "
              f"{r['recall']:<10.2f} {r['precision']:<10.2f} {r['f1']:<10.2f} "
              f"{r['tp']:<6} {r['fp']:<6} {r['rejected']:<7}")

    # Mejor configuración
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)

    best = results[0]
    print(f"\nParameters:")
    print(f"  delta_z_threshold = {best['delta_z']}m")
    print(f"  kdtree_radius = {best['radius']}m")
    print(f"  min_neighbors = {best['min_neighbors']}")

    print(f"\nPerformance:")
    print(f"  Recall:    {best['recall']:.2f}% ({best['tp']} / {best['targets']} targets detected)")
    print(f"  Precision: {best['precision']:.2f}% ({best['tp']} / {best['rejected']} correct)")
    print(f"  F1-score:  {best['f1']:.2f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {best['tp']}")
    print(f"  False Positives: {best['fp']}")
    print(f"  False Negatives: {best['targets'] - best['tp']}")

    # Análisis de impacto de parámetros
    print("\n" + "=" * 100)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 100)

    # Agrupar por delta_z
    print("\nImpact of delta_z_threshold:")
    for dz in delta_z_vals:
        subset = [r for r in results if r['delta_z'] == dz]
        avg_recall = np.mean([r['recall'] for r in subset])
        avg_prec = np.mean([r['precision'] for r in subset])
        avg_f1 = np.mean([r['f1'] for r in subset])
        print(f"  δZ={dz:.2f}m: Avg Recall={avg_recall:.1f}%, Prec={avg_prec:.1f}%, F1={avg_f1:.1f}")

    # Agrupar por radius
    print("\nImpact of kdtree_radius:")
    for rad in radius_vals:
        subset = [r for r in results if r['radius'] == rad]
        avg_recall = np.mean([r['recall'] for r in subset])
        avg_prec = np.mean([r['precision'] for r in subset])
        avg_f1 = np.mean([r['f1'] for r in subset])
        print(f"  r={rad:.1f}m: Avg Recall={avg_recall:.1f}%, Prec={avg_prec:.1f}%, F1={avg_f1:.1f}")

    # Agrupar por min_neighbors
    print("\nImpact of min_neighbors:")
    for mn in min_neigh_vals:
        subset = [r for r in results if r['min_neighbors'] == mn]
        avg_recall = np.mean([r['recall'] for r in subset])
        avg_prec = np.mean([r['precision'] for r in subset])
        avg_f1 = np.mean([r['f1'] for r in subset])
        print(f"  min_n={mn}: Avg Recall={avg_recall:.1f}%, Prec={avg_prec:.1f}%, F1={avg_f1:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parameter sweep for wall rejection")
    parser.add_argument("--scan", type=int, default=0, help="Scan ID")
    args = parser.parse_args()

    parameter_sweep(args.scan)
