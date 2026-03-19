#!/usr/bin/env python3
"""
Script standalone para debuggear Patchwork++ sin dependencias de ROS 2.
Muestra cuántos planos con normales horizontales (paredes) genera Patchwork++.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.spatial import cKDTree

# Añadir Patchwork++ al path (desde el entorno virtual pwenv)

def main():
    # Importar Patchwork++
    try:
        import pypatchworkpp
        print("✅ pypatchworkpp importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando pypatchworkpp: {e}")
        sys.exit(1)

    # Configurar Patchwork++ (mismo que paso_1.py)
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.sensor_height = 1.73
    params.min_range = 2.7
    params.max_range = 80.0
    params.num_iter = 3
    params.num_lpr = 20
    params.num_min_pts = 10
    params.th_dist = 0.2
    params.uprightness_thr = 0.707
    params.adaptive_seed_selection_margin = -1.1
    params.enable_RNR = False

    # CZM
    params.num_zones = 4
    params.num_rings_each_zone = [2, 4, 4, 4]
    params.num_sectors_each_zone = [16, 32, 54, 32]

    patchwork = pypatchworkpp.patchworkpp(params)
    print(f"✅ Patchwork++ inicializado")

    # Cargar datos
    bin_file = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/data/000000.bin")
    if not bin_file.exists():
        print(f"❌ Archivo no encontrado: {bin_file}")
        sys.exit(1)

    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    print(f"✅ Cargados {len(points)} puntos desde {bin_file.name}")

    # Ejecutar Patchwork++
    patchwork.estimateGround(points)
    ground_points = patchwork.getGround()
    centers = patchwork.getCenters()
    normals = patchwork.getNormals()

    print(f"\n📊 RESULTADOS DE PATCHWORK++:")
    print(f"   - Puntos clasificados como suelo: {len(ground_points)}/{len(points)} ({100*len(ground_points)/len(points):.1f}%)")
    print(f"   - Planos generados (bins CZM): {len(centers)}")

    # Analizar normales
    print(f"\n🔍 ANÁLISIS DE NORMALES:")

    nz_values = [n[2] if n[2] > 0 else -n[2] for n in normals]  # Asegurar positivas

    # Histograma de normales
    bins_hist = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    counts, _ = np.histogram(nz_values, bins=bins_hist)

    print(f"   Distribución de componente vertical (nz) de las normales:")
    labels = ["0.0-0.3 (muy horizontal)", "0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95-1.0 (muy vertical)"]
    for i, (label, count) in enumerate(zip(labels, counts)):
        pct = 100 * count / len(normals) if len(normals) > 0 else 0
        print(f"      {label:25s}: {count:4d} planos ({pct:5.1f}%)")

    # Detectar planos sospechosos (paredes)
    threshold_wall = 0.7
    suspicious_planes = []

    for i, (c, n) in enumerate(zip(centers, normals)):
        nz = abs(n[2])  # Asegurar positivo
        if nz < threshold_wall:
            suspicious_planes.append((i, c, n, nz))

    print(f"\n⚠️  PLANOS SOSPECHOSOS (nz < {threshold_wall}):")
    print(f"   Total: {len(suspicious_planes)}/{len(centers)} planos ({100*len(suspicious_planes)/len(centers):.1f}%)")

    if len(suspicious_planes) > 0:
        print(f"\n   Primeros 10 planos sospechosos:")
        print(f"   {'ID':>4} | {'X':>7} {'Y':>7} {'Z':>7} | {'nx':>6} {'ny':>6} {'nz':>6}")
        print(f"   {'-'*4}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*6}-{'-'*6}-{'-'*6}")

        for idx, (i, c, n, nz) in enumerate(suspicious_planes[:10]):
            print(f"   {i:4d} | {c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f} | {n[0]:6.3f} {n[1]:6.3f} {n[2]:6.3f}")

        # Análisis de altura de centroides sospechosos
        z_suspicious = [c[2] for _, c, _, _ in suspicious_planes]
        print(f"\n   Altura (Z) de centroides sospechosos:")
        print(f"      Min: {np.min(z_suspicious):.2f}m, Max: {np.max(z_suspicious):.2f}m, Media: {np.mean(z_suspicious):.2f}m")

    # Análisis con KDTree (validación geométrica)
    if len(suspicious_planes) > 0:
        print(f"\n🔬 VALIDACIÓN GEOMÉTRICA (Delta Z local):")

        try:
            ground_tree = cKDTree(ground_points)

            confirmed_walls = []
            false_positives = []  # Rampas/bordillos

            for i, c, n, nz in suspicious_planes[:20]:  # Analizar primeros 20
                # Buscar puntos de suelo en 0.5m alrededor del centroide
                idx = ground_tree.query_ball_point(c, r=0.5)

                if len(idx) > 5:
                    local_pts = ground_points[idx]
                    z_high = np.percentile(local_pts[:, 2], 95)
                    z_low = np.percentile(local_pts[:, 2], 5)
                    delta_z = z_high - z_low

                    if delta_z > 0.3:  # Threshold para pared
                        confirmed_walls.append((i, c, n, delta_z))
                    else:
                        false_positives.append((i, c, n, delta_z))

            print(f"   De {min(20, len(suspicious_planes))} planos analizados:")
            print(f"      ✅ PAREDES confirmadas (ΔZ > 0.3m): {len(confirmed_walls)}")
            print(f"      ⚪ Rampas/Bordillos (ΔZ < 0.3m):  {len(false_positives)}")

            if len(confirmed_walls) > 0:
                print(f"\n   Ejemplos de PAREDES confirmadas:")
                print(f"   {'ID':>4} | {'Centroide Z':>12} | {'nz':>6} | {'ΔZ local':>10}")
                print(f"   {'-'*4}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}")
                for i, c, n, dz in confirmed_walls[:5]:
                    print(f"   {i:4d} | {c[2]:12.2f} | {n[2]:6.3f} | {dz:10.3f}m")

            if len(false_positives) > 0:
                print(f"\n   Ejemplos de RAMPAS/BORDILLOS (no son paredes):")
                print(f"   {'ID':>4} | {'Centroide Z':>12} | {'nz':>6} | {'ΔZ local':>10}")
                print(f"   {'-'*4}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}")
                for i, c, n, dz in false_positives[:5]:
                    print(f"   {i:4d} | {c[2]:12.2f} | {n[2]:6.3f} | {dz:10.3f}m")

        except Exception as e:
            print(f"   ❌ Error en análisis KDTree: {e}")

    print(f"\n{'='*80}")
    print(f"CONCLUSIÓN:")
    print(f"{'='*80}")
    print(f"Patchwork++ genera planos para TODOS los bins CZM que tienen puntos,")
    print(f"incluyendo bins que contienen paredes/obstáculos verticales.")
    print(f"")
    print(f"⚠️  {len(suspicious_planes)} de {len(centers)} planos tienen normales horizontales (nz < 0.7)")
    print(f"")
    print(f"Por lo tanto, DEBES implementar RECHAZO POST-PROCESADO de planos con:")
    print(f"  1. Normal horizontal (nz < 0.7)")
    print(f"  2. Alta variación de altura local (ΔZ > 0.3m)")
    print(f"")
    print(f"Estos planos corresponden a PAREDES, no suelo, y deben ser rechazados.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
