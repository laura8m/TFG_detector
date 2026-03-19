#!/usr/bin/env python3
"""
Test script para validar wall rejection v2.1 (point-wise analysis)

Compara:
  - V2.0 (bin-wise): Analiza bins completos con normal threshold
  - V2.1 (point-wise): Analiza cada punto individualmente

Dataset: KITTI SemanticKITTI con ground truth de paredes
"""

import sys
import os

# Añadir paths necesarios
patchwork_site_packages = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_site_packages)
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')

# Cambiar al directorio correcto para imports relativos
os.chdir('/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')

import numpy as np
import pypatchworkpp
from ring_anomaly_detection import (
    estimate_local_ground_planes,
    _validate_and_reject_walls,
    _validate_and_reject_walls_pointwise
)


def load_kitti_bin(file_path):
    """Carga archivo .bin de KITTI"""
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]  # (x, y, z, intensity) -> (x, y, z)
    return points


def load_kitti_labels(file_path):
    """Carga labels de SemanticKITTI"""
    labels = np.fromfile(file_path, dtype=np.uint32)
    labels = labels & 0xFFFF  # Máscara para obtener semantic label
    return labels


def analyze_wall_rejection(scan_id=0):
    """
    Analiza wall rejection con ground truth de SemanticKITTI

    Args:
        scan_id (int): ID del scan a analizar (0-4540 en seq 00)
    """
    # Rutas de datos (data_kitti sequence 04)
    base_velodyne = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne"
    base_labels = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels"
    scan_file = f"{base_velodyne}/{scan_id:06d}.bin"
    label_file = f"{base_labels}/{scan_id:06d}.label"

    print(f"=" * 80)
    print(f"WALL REJECTION TEST - Scan {scan_id:06d}")
    print(f"=" * 80)

    # Cargar datos
    points = load_kitti_bin(scan_file)
    labels = load_kitti_labels(label_file)

    print(f"\nPoints: {len(points)}")
    print(f"Labels: {len(labels)}")
    print(f"GT labels unique: {np.unique(labels)}")

    # Identificar wall points en ground truth (classes 50=building, 51=fence, 52=other-structure)
    wall_classes = [50, 51, 52]
    gt_wall_mask = np.isin(labels, wall_classes)
    gt_wall_indices = np.where(gt_wall_mask)[0]

    # Ground truth (road=40, sidewalk=48, parking=44, other-ground=49, terrain=72)
    ground_classes = [40, 44, 48, 49, 72]
    gt_ground_mask = np.isin(labels, ground_classes)
    gt_ground_indices = np.where(gt_ground_mask)[0]

    print(f"\n--- GROUND TRUTH ---")
    print(f"GT wall points (classes {wall_classes}): {len(gt_wall_indices)}")
    print(f"GT ground points (classes {ground_classes}): {len(gt_ground_indices)}")

    # Ejecutar Patchwork++
    print(f"\n--- PATCHWORK++ BASELINE ---")

    # Inicializar Patchwork++
    pw_params = pypatchworkpp.Parameters()
    pw_params.verbose = False
    pw = pypatchworkpp.patchworkpp(pw_params)

    result_baseline = estimate_local_ground_planes(
        points,
        pw,
        patchwork_params=pw_params,
        enable_wall_rejection=False  # Sin wall rejection
    )

    ground_pw = result_baseline['ground_indices']
    nonground_pw = result_baseline['nonground_indices']
    local_planes = result_baseline['local_planes']

    print(f"Patchwork++ ground: {len(ground_pw)}, non-ground: {len(nonground_pw)}")

    # Analizar falsos positivos (wall points clasificados como ground)
    fp_wall_mask = np.isin(ground_pw, gt_wall_indices)
    fp_wall_indices = ground_pw[fp_wall_mask]

    print(f"GT wall points classified as GROUND by Patchwork++: {len(fp_wall_indices)}")
    print(f"  -> These are the targets for wall rejection")

    if len(fp_wall_indices) == 0:
        print("\n✅ Patchwork++ no clasificó mal ninguna pared. No hay nada que rechazar.")
        return

    # ====================
    # V2.0: BIN-WISE (legacy)
    # ====================
    print(f"\n--- V2.0: BIN-WISE WALL REJECTION (legacy) ---")

    rejected_v20 = _validate_and_reject_walls(
        points,
        local_planes,
        ground_pw,
        normal_threshold=0.7,
        delta_z_threshold=0.3,
        use_kdtree=True,
        use_percentiles=True,
        use_height_fallback=True,
        kdtree_radius=0.5,
        min_neighbors=5,
        height_fallback_z=-1.0
    )

    # Calcular true positives (wall points correctamente rechazados)
    tp_v20 = np.intersect1d(rejected_v20, gt_wall_indices)
    # False positives (ground points incorrectamente rechazados)
    fp_v20 = np.intersect1d(rejected_v20, gt_ground_indices)

    print(f"Total rejected: {len(rejected_v20)}")
    print(f"  ✓ True Positives (actual walls): {len(tp_v20)} / {len(fp_wall_indices)}")
    print(f"  ✗ False Positives (actual ground): {len(fp_v20)}")

    recall_v20 = len(tp_v20) / len(fp_wall_indices) * 100 if len(fp_wall_indices) > 0 else 0
    precision_v20 = len(tp_v20) / len(rejected_v20) * 100 if len(rejected_v20) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Recall:    {recall_v20:.2f}% (de {len(fp_wall_indices)} wall points mal clasificados)")
    print(f"  Precision: {precision_v20:.2f}% (de {len(rejected_v20)} rechazos)")

    # ====================
    # V2.1: POINT-WISE (nuevo)
    # ====================
    print(f"\n--- V2.1: POINT-WISE WALL REJECTION (nuevo) ---")

    rejected_v21 = _validate_and_reject_walls_pointwise(
        points,
        ground_pw,
        delta_z_threshold=0.3,
        use_percentiles=True,
        kdtree_radius=0.5,
        min_neighbors=5
    )

    # Calcular métricas
    tp_v21 = np.intersect1d(rejected_v21, gt_wall_indices)
    fp_v21 = np.intersect1d(rejected_v21, gt_ground_indices)

    print(f"Total rejected: {len(rejected_v21)}")
    print(f"  ✓ True Positives (actual walls): {len(tp_v21)} / {len(fp_wall_indices)}")
    print(f"  ✗ False Positives (actual ground): {len(fp_v21)}")

    recall_v21 = len(tp_v21) / len(fp_wall_indices) * 100 if len(fp_wall_indices) > 0 else 0
    precision_v21 = len(tp_v21) / len(rejected_v21) * 100 if len(rejected_v21) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Recall:    {recall_v21:.2f}% (de {len(fp_wall_indices)} wall points mal clasificados)")
    print(f"  Precision: {precision_v21:.2f}% (de {len(rejected_v21)} rechazos)")

    # ====================
    # COMPARACIÓN
    # ====================
    print(f"\n{'=' * 80}")
    print(f"COMPARISON: V2.0 vs V2.1")
    print(f"{'=' * 80}")

    print(f"\n{'Metric':<20} {'V2.0 (bin-wise)':<20} {'V2.1 (point-wise)':<20} {'Δ':<15}")
    print(f"{'-' * 80}")
    print(f"{'Rejected':<20} {len(rejected_v20):<20} {len(rejected_v21):<20} "
          f"{'+' if len(rejected_v21) > len(rejected_v20) else ''}{len(rejected_v21) - len(rejected_v20)}")
    print(f"{'True Positives':<20} {len(tp_v20):<20} {len(tp_v21):<20} "
          f"{'+' if len(tp_v21) > len(tp_v20) else ''}{len(tp_v21) - len(tp_v20)}")
    print(f"{'False Positives':<20} {len(fp_v20):<20} {len(fp_v21):<20} "
          f"{'+' if len(fp_v21) > len(fp_v20) else ''}{len(fp_v21) - len(fp_v20)}")
    print(f"{'Recall (%)':<20} {recall_v20:<20.2f} {recall_v21:<20.2f} "
          f"{'+' if recall_v21 > recall_v20 else ''}{recall_v21 - recall_v20:.2f}")
    print(f"{'Precision (%)':<20} {precision_v20:<20.2f} {precision_v21:<20.2f} "
          f"{'+' if precision_v21 > precision_v20 else ''}{precision_v21 - precision_v20:.2f}")

    # Conclusiones
    print(f"\n{'=' * 80}")
    print(f"CONCLUSIONES")
    print(f"{'=' * 80}")

    if len(rejected_v21) > len(rejected_v20):
        print(f"✅ V2.1 detectó {len(rejected_v21) - len(rejected_v20)} más puntos de pared")
    elif len(rejected_v21) == len(rejected_v20) == 0:
        print(f"⚠️  Ambas versiones fallaron en detectar paredes")
        print(f"    Posibles causas:")
        print(f"    - ΔZ threshold muy alto (actual: 0.3m)")
        print(f"    - Radius muy pequeño (actual: 0.5m)")
        print(f"    - Paredes con gradiente suave (no escalonado)")
    else:
        print(f"ℹ️  Ambas versiones rechazaron cantidades similares")

    if recall_v21 > recall_v20:
        print(f"✅ V2.1 mejoró recall en {recall_v21 - recall_v20:.2f}%")
    elif recall_v21 == recall_v20 == 0:
        print(f"⚠️  Recall = 0% para ambas (no detectaron ninguna pared)")
    else:
        print(f"⚠️  V2.1 redujo recall (puede necesitar ajuste de parámetros)")

    if precision_v21 > precision_v20:
        print(f"✅ V2.1 mejoró precisión en {precision_v21 - precision_v20:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test wall rejection v2.1")
    parser.add_argument("--scan", type=int, default=0, help="Scan ID (0-4540)")
    args = parser.parse_args()

    analyze_wall_rejection(args.scan)
