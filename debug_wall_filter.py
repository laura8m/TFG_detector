#!/usr/bin/env python3
"""
Debug detallado del filtro geométrico de paredes.
Muestra por qué los planos candidatos no se rechazan.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from data_paths import get_scan_file

# Importar Patchwork++ directamente
import pypatchworkpp

def debug_wall_filter(scan_id=0, sequence='00'):
    """Debug detallado del filtro geométrico"""

    # Cargar scan
    scan_path = get_scan_file(sequence, scan_id)
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]

    print(f"\n{'='*80}")
    print(f"DEBUG FILTRO GEOMÉTRICO - Scan {scan_id:06d} (Seq {sequence})")
    print(f"{'='*80}\n")

    # Inicializar Patchwork++
    params = pypatchworkpp.Parameters()
    params.verbose = False
    patchwork = pypatchworkpp.patchworkpp(params)

    # Ejecutar segmentación
    patchwork.estimateGround(points)
    ground_idx = patchwork.getGroundIndices()
    ground_points = points[ground_idx]

    normals = patchwork.getNormals()
    centers = patchwork.getCenters()

    print(f"Ground points: {len(ground_points)}")
    print(f"Planos detectados: {len(centers)}")

    # KDTree para búsqueda de vecinos
    if len(ground_points) > 0:
        ground_tree = cKDTree(ground_points)
    else:
        print("ERROR: No hay ground points")
        return

    # Parámetros del filtro
    wall_slope = 0.75
    wall_delta_z = 0.15
    wall_radius = 0.8

    print(f"\nParámetros del filtro:")
    print(f"  wall_rejection_slope: {wall_slope}")
    print(f"  wall_height_diff_threshold: {wall_delta_z}m")
    print(f"  wall_kdtree_radius: {wall_radius}m")

    # Analizar cada plano
    print(f"\n{'='*80}")
    print("ANÁLISIS POR PLANO:")
    print(f"{'='*80}\n")

    candidates_analyzed = 0
    walls_detected = 0

    for i in range(len(centers)):
        c = centers[i]
        n = normals[i]
        nz = abs(n[2])

        # Solo analizar candidatos (nz < 0.75)
        if nz >= wall_slope:
            continue

        candidates_analyzed += 1
        angle = np.degrees(np.arccos(nz))

        print(f"Plano {i}: nz={nz:.3f}, ángulo={angle:.1f}°")
        print(f"  Centro: {c}")
        print(f"  Normal: {n}")

        # Buscar vecinos
        idx = ground_tree.query_ball_point(c, r=wall_radius)
        print(f"  Vecinos encontrados en r={wall_radius}m: {len(idx)}")

        if len(idx) < 5:
            print(f"  ❌ Rechazado: pocos vecinos (< 5)")
            print()
            continue

        # Calcular delta_z
        z_vals = ground_points[idx, 2]
        z_high = np.percentile(z_vals, 95)
        z_low = np.percentile(z_vals, 5)
        delta_z = z_high - z_low

        print(f"  Alturas vecinos:")
        print(f"    min: {np.min(z_vals):.3f}m, max: {np.max(z_vals):.3f}m")
        print(f"    p05: {z_low:.3f}m, p95: {z_high:.3f}m")
        print(f"    delta_z (p95-p05): {delta_z:.3f}m")

        if delta_z > wall_delta_z:
            print(f"  ✓ PARED DETECTADA (delta_z={delta_z:.3f}m > {wall_delta_z}m)")
            walls_detected += 1
        else:
            print(f"  ❌ Rechazado: delta_z={delta_z:.3f}m ≤ {wall_delta_z}m")

        print()

    print(f"{'='*80}")
    print("RESUMEN:")
    print(f"{'='*80}")
    print(f"Planos candidatos analizados: {candidates_analyzed}")
    print(f"Paredes detectadas: {walls_detected}")

    if candidates_analyzed > 0 and walls_detected == 0:
        print(f"\n⚠️  PROBLEMA: Ningún candidato pasó el filtro geométrico")
        print(f"\nPosibles soluciones:")
        print(f"  1. Reducir wall_height_diff_threshold de {wall_delta_z}m a 0.10m")
        print(f"  2. Reducir wall_kdtree_radius de {wall_radius}m a 0.5m")
        print(f"     (ventana más pequeña = delta_z más local)")
        print(f"  3. Cambiar criterio a usar std(z) en lugar de percentiles")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debug wall geometric filter')
    parser.add_argument('--scan', type=int, default=0, help='Scan ID')
    parser.add_argument('--sequence', type=str, default='00', help='KITTI sequence')
    args = parser.parse_args()

    debug_wall_filter(args.scan, args.sequence)
