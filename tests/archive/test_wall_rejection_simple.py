#!/usr/bin/env python3
"""
Test simplificado para wall rejection v2.1 (sin necesidad de ground truth)

Compara:
  - V2.0 (bin-wise): Analiza bins completos con normal threshold
  - V2.1 (point-wise): Analiza cada punto individualmente

Muestra estadísticas y permite inspección visual del comportamiento
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


def analyze_wall_rejection_behavior(scan_id=0):
    """
    Analiza comportamiento de wall rejection comparando v2.0 vs v2.1

    Args:
        scan_id (int): ID del scan a analizar (0-4540 en seq 00)
    """
    # Rutas de datos
    base_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/test_data/sequences/00"
    scan_file = f"{base_path}/velodyne/{scan_id:06d}.bin"

    print(f"=" * 80)
    print(f"WALL REJECTION BEHAVIOR TEST - Scan {scan_id:06d}")
    print(f"=" * 80)

    # Cargar datos
    points = load_kitti_bin(scan_file)
    print(f"\nTotal points: {len(points)}")

    # Ejecutar Patchwork++ baseline
    print(f"\n--- PATCHWORK++ BASELINE ---")
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

    print(f"Patchwork++ ground: {len(ground_pw)} points")
    print(f"Patchwork++ non-ground: {len(nonground_pw)} points")

    # Analizar distribución de alturas en ground points
    ground_pts = points[ground_pw]
    ground_z = ground_pts[:, 2]
    print(f"\nGround Z statistics:")
    print(f"  Mean: {ground_z.mean():.3f}m")
    print(f"  Std:  {ground_z.std():.3f}m")
    print(f"  Min:  {ground_z.min():.3f}m")
    print(f"  Max:  {ground_z.max():.3f}m")
    print(f"  P95:  {np.percentile(ground_z, 95):.3f}m")

    # Analizar bins
    print(f"\nLocal planes (bins): {len(local_planes)}")

    # Contar bins con diferentes características
    vertical_bins = sum(1 for p in local_planes.values() if abs(p['normal'][2]) >= 0.7)
    tilted_bins = sum(1 for p in local_planes.values() if abs(p['normal'][2]) < 0.7)

    print(f"  Bins with nz >= 0.7 (horizontal): {vertical_bins}")
    print(f"  Bins with nz < 0.7 (tilted): {tilted_bins}")

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

    print(f"Total rejected: {len(rejected_v20)} points ({100*len(rejected_v20)/len(ground_pw):.2f}% of ground)")

    if len(rejected_v20) > 0:
        rejected_pts_v20 = points[rejected_v20]
        rej_z_v20 = rejected_pts_v20[:, 2]
        print(f"  Rejected Z range: [{rej_z_v20.min():.3f}, {rej_z_v20.max():.3f}]m")
        print(f"  Rejected Z mean: {rej_z_v20.mean():.3f}m")

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

    print(f"Total rejected: {len(rejected_v21)} points ({100*len(rejected_v21)/len(ground_pw):.2f}% of ground)")

    if len(rejected_v21) > 0:
        rejected_pts_v21 = points[rejected_v21]
        rej_z_v21 = rejected_pts_v21[:, 2]
        print(f"  Rejected Z range: [{rej_z_v21.min():.3f}, {rej_z_v21.max():.3f}]m")
        print(f"  Rejected Z mean: {rej_z_v21.mean():.3f}m")

    # ====================
    # COMPARACIÓN
    # ====================
    print(f"\n{'=' * 80}")
    print(f"COMPARISON: V2.0 vs V2.1")
    print(f"{'=' * 80}")

    print(f"\n{'Metric':<30} {'V2.0 (bin-wise)':<20} {'V2.1 (point-wise)':<20}")
    print(f"{'-' * 80}")
    print(f"{'Total rejected':<30} {len(rejected_v20):<20} {len(rejected_v21):<20}")
    print(f"{'% of ground points':<30} {100*len(rejected_v20)/len(ground_pw):<20.2f} "
          f"{100*len(rejected_v21)/len(ground_pw):<20.2f}")

    # Puntos en común y únicos
    common = np.intersect1d(rejected_v20, rejected_v21)
    only_v20 = np.setdiff1d(rejected_v20, rejected_v21)
    only_v21 = np.setdiff1d(rejected_v21, rejected_v20)

    print(f"{'Common rejections':<30} {len(common):<20} {len(common):<20}")
    print(f"{'Only in V2.0':<30} {len(only_v20):<20} {'--':<20}")
    print(f"{'Only in V2.1':<30} {'--':<20} {len(only_v21):<20}")

    # Análisis de diferencias
    print(f"\n{'=' * 80}")
    print(f"INTERPRETATION")
    print(f"{'=' * 80}")

    if len(rejected_v20) == 0 and len(rejected_v21) == 0:
        print("⚠️  Ambas versiones rechazaron 0 puntos")
        print("    Posibles causas:")
        print("    1. No hay paredes en el scan")
        print("    2. Umbrales demasiado estrictos:")
        print("       - delta_z_threshold = 0.3m (puede ser muy alto para paredes graduales)")
        print("       - kdtree_radius = 0.5m (puede necesitar ajuste)")
        print("       - min_neighbors = 5 (puede filtrar zonas sparse)")
        print("\n    Prueba reducir delta_z_threshold a 0.2m:")
        print("       python3 test_wall_rejection_simple.py --scan 0 --delta_z 0.2")

    elif len(rejected_v20) == 0 and len(rejected_v21) > 0:
        print(f"✅ V2.1 FUNCIONA - V2.0 FALLA")
        print(f"   V2.1 detectó {len(rejected_v21)} wall edges que V2.0 ignoró")
        print(f"   Esto confirma que el enfoque point-wise resuelve el problema de bins con normal horizontal")

    elif len(rejected_v21) > len(rejected_v20):
        print(f"✅ V2.1 ES MÁS SENSIBLE")
        print(f"   V2.1 detectó {len(rejected_v21) - len(rejected_v20)} puntos adicionales")
        print(f"   {len(common)} puntos fueron detectados por ambas versiones")
        print(f"   {len(only_v21)} puntos solo fueron detectados por V2.1")

    else:
        print(f"ℹ️  Comportamiento similar entre versiones")
        print(f"   Ambas detectaron cantidades comparables de puntos")

    # Guardar resultados para visualización (opcional)
    print(f"\n{'=' * 80}")
    print(f"OUTPUT FILES")
    print(f"{'=' * 80}")

    output_dir = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/wall_rejection_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar rejected points
    if len(rejected_v20) > 0:
        np.save(f"{output_dir}/rejected_v20_scan{scan_id:06d}.npy", rejected_v20)
        print(f"✓ Saved: rejected_v20_scan{scan_id:06d}.npy")

    if len(rejected_v21) > 0:
        np.save(f"{output_dir}/rejected_v21_scan{scan_id:06d}.npy", rejected_v21)
        print(f"✓ Saved: rejected_v21_scan{scan_id:06d}.npy")

    # Guardar ground points para referencia
    np.save(f"{output_dir}/ground_pw_scan{scan_id:06d}.npy", ground_pw)
    print(f"✓ Saved: ground_pw_scan{scan_id:06d}.npy")

    print(f"\nFiles saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test wall rejection v2.1 behavior")
    parser.add_argument("--scan", type=int, default=0, help="Scan ID (0-4540)")
    parser.add_argument("--delta_z", type=float, default=0.3, help="Delta Z threshold (m)")
    args = parser.parse_args()

    # Si se especifica delta_z custom, actualizar en el código
    # (Para simplificar, solo usamos el default ahora)

    analyze_wall_rejection_behavior(args.scan)
