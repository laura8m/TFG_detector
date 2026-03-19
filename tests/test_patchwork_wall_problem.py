#!/usr/bin/env python3
"""
Test: Verificar si Patchwork++ clasifica paredes/obstaculos como suelo.

Usa SemanticKITTI labels como ground truth para comprobar cuantos puntos
que SON obstaculos (edificios, vallas, vegetacion, vehiculos) quedan
clasificados como "ground" por Patchwork++.

Dataset: KITTI Sequence 04, Frame 000000
"""

import numpy as np
import sys
import os


import pypatchworkpp

# ============================================================
# Configuracion
# ============================================================
SCAN_FILE = '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne/000000.bin'
LABEL_FILE = '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels/000000.label'

# SemanticKITTI label mapping
LABEL_NAMES = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle',
}

# Labels que son GROUND real
GROUND_LABELS = {40, 44, 48, 49, 60, 72}  # road, parking, sidewalk, other-ground, lane-marking, terrain

# Labels que son OBSTACULOS reales (NO deberian ser ground)
OBSTACLE_LABELS = {
    10, 11, 13, 15, 16, 18,  # vehicles
    20,                       # other-vehicle
    30, 31, 32,               # person, bicyclist, motorcyclist
    50, 51, 52,               # building, fence, other-structure
    70, 71,                   # vegetation, trunk
    80, 81,                   # pole, traffic-sign
    99,                       # other-object
    252, 253, 254, 255, 256, 257, 258, 259  # moving objects
}

# ============================================================
# Cargar datos
# ============================================================
print("=" * 80)
print("TEST: Patchwork++ clasifica paredes/obstaculos como suelo?")
print("=" * 80)

# Cargar point cloud
points_raw = np.fromfile(SCAN_FILE, dtype=np.float32).reshape(-1, 4)
points = points_raw[:, :3]
print(f"\nPoint cloud: {len(points)} puntos")

# Cargar labels
labels_raw = np.fromfile(LABEL_FILE, dtype=np.uint32)
semantic_labels = labels_raw & 0xFFFF
print(f"Labels: {len(semantic_labels)} etiquetas")

# Estadisticas GT
gt_ground_mask = np.zeros(len(semantic_labels), dtype=bool)
for l in GROUND_LABELS:
    gt_ground_mask |= (semantic_labels == l)

gt_obstacle_mask = np.zeros(len(semantic_labels), dtype=bool)
for l in OBSTACLE_LABELS:
    gt_obstacle_mask |= (semantic_labels == l)

gt_unlabeled_mask = ~(gt_ground_mask | gt_obstacle_mask)

print(f"\nGround truth (SemanticKITTI):")
print(f"  Ground real:    {np.sum(gt_ground_mask):>7d} ({100*np.mean(gt_ground_mask):.1f}%)")
print(f"  Obstaculos:     {np.sum(gt_obstacle_mask):>7d} ({100*np.mean(gt_obstacle_mask):.1f}%)")
print(f"  Unlabeled/otro: {np.sum(gt_unlabeled_mask):>7d} ({100*np.mean(gt_unlabeled_mask):.1f}%)")

# ============================================================
# Ejecutar Patchwork++
# ============================================================
print("\n" + "-" * 80)
print("Ejecutando Patchwork++ (configuracion por defecto)...")
print("-" * 80)

params = pypatchworkpp.Parameters()
params.verbose = False

pwpp = pypatchworkpp.patchworkpp(params)
pwpp.estimateGround(points)

ground_pts = pwpp.getGround()
nonground_pts = pwpp.getNonground()

print(f"\nPatchwork++ resultado:")
print(f"  Ground:     {len(ground_pts):>7d} puntos")
print(f"  Non-ground: {len(nonground_pts):>7d} puntos")

# Reconstruir indices usando KDTree
from scipy.spatial import cKDTree

tree = cKDTree(points)

# Encontrar indices de ground points
if len(ground_pts) > 0:
    _, pw_ground_indices = tree.query(ground_pts, k=1)
else:
    pw_ground_indices = np.array([], dtype=int)

pw_ground_mask = np.zeros(len(points), dtype=bool)
pw_ground_mask[pw_ground_indices] = True

# ============================================================
# Analisis: Obstaculos clasificados como ground
# ============================================================
print("\n" + "=" * 80)
print("ANALISIS: Obstaculos MAL clasificados como ground por Patchwork++")
print("=" * 80)

# Puntos que Patchwork++ dice ground PERO son obstaculos segun GT
misclassified_mask = pw_ground_mask & gt_obstacle_mask
n_misclassified = np.sum(misclassified_mask)
misclassified_indices = np.where(misclassified_mask)[0]

print(f"\nObstaculos clasificados como ground: {n_misclassified}")
print(f"  De un total de {np.sum(gt_obstacle_mask)} obstaculos GT")
print(f"  Porcentaje: {100 * n_misclassified / max(1, np.sum(gt_obstacle_mask)):.2f}%")

if n_misclassified > 0:
    # Desglose por tipo de obstaculo
    print(f"\nDesglose por tipo de obstaculo mal clasificado como ground:")
    print(f"  {'Label':<25s} {'Count':>7s} {'% del total':>12s}")
    print(f"  {'-'*25} {'-'*7} {'-'*12}")

    misclassified_labels = semantic_labels[misclassified_indices]
    unique_labels, counts = np.unique(misclassified_labels, return_counts=True)

    # Ordenar por count descendente
    sort_idx = np.argsort(-counts)
    for idx in sort_idx:
        label = unique_labels[idx]
        count = counts[idx]
        name = LABEL_NAMES.get(label, f'unknown-{label}')
        total_of_this = np.sum(semantic_labels == label)
        pct = 100 * count / max(1, total_of_this)
        print(f"  {name:<25s} {count:>7d} ({pct:.1f}% de {total_of_this} totales)")

# ============================================================
# Analisis de normales de los planos
# ============================================================
print("\n" + "=" * 80)
print("ANALISIS: Normales de los planos CZM de Patchwork++")
print("=" * 80)

centers = pwpp.getCenters()
normals = pwpp.getNormals()

if len(normals) > 0:
    normals = np.array(normals)
    nz_vals = np.abs(normals[:, 2])  # Componente vertical

    print(f"\nTotal planos: {len(normals)}")
    print(f"\nDistribucion de |nz| (componente vertical de la normal):")
    bins_nz = [(0, 0.3, "MUY inclinado (pared)"),
               (0.3, 0.5, "Inclinado"),
               (0.5, 0.7, "Moderado"),
               (0.7, 0.85, "Casi horizontal"),
               (0.85, 0.95, "Horizontal"),
               (0.95, 1.01, "Suelo perfecto")]

    for lo, hi, desc in bins_nz:
        mask = (nz_vals >= lo) & (nz_vals < hi)
        n = np.sum(mask)
        marker = " *** SOSPECHOSO" if lo < 0.7 else ""
        print(f"  nz [{lo:.2f}, {hi:.2f}): {n:>4d} planos ({100*n/len(normals):>5.1f}%) - {desc}{marker}")

    n_vertical = np.sum(nz_vals < 0.7)
    print(f"\n  TOTAL planos verticales (nz < 0.7): {n_vertical}/{len(normals)} ({100*n_vertical/len(normals):.1f}%)")
else:
    print("  No se obtuvieron normales")

# ============================================================
# Analisis de puntos ground con Z alta
# ============================================================
print("\n" + "=" * 80)
print("ANALISIS: Puntos ground con altura anomala")
print("=" * 80)

if len(pw_ground_indices) > 0:
    ground_z = points[pw_ground_indices, 2]

    print(f"\nAltura Z de puntos ground:")
    print(f"  Min:  {ground_z.min():.2f} m")
    print(f"  Max:  {ground_z.max():.2f} m")
    print(f"  Mean: {ground_z.mean():.2f} m")
    print(f"  Std:  {ground_z.std():.2f} m")

    # Puntos ground con Z > 0 (por encima del sensor)
    high_ground = ground_z > 0.0
    print(f"\n  Puntos ground con Z > 0.0m (por encima del sensor): {np.sum(high_ground)}")

    # Puntos ground con Z > 0.5m
    very_high_ground = ground_z > 0.5
    print(f"  Puntos ground con Z > 0.5m: {np.sum(very_high_ground)}")

    # Variacion vertical en clusters
    from scipy.spatial import cKDTree as KDT

    # Analizar variacion vertical en vecindades de 1m
    ground_pts_3d = points[pw_ground_indices]
    ground_xy = ground_pts_3d[:, :2]  # Solo X,Y

    if len(ground_xy) > 100:
        tree_xy = KDT(ground_xy)
        # Muestrear 2000 puntos aleatorios
        sample_size = min(2000, len(ground_xy))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(ground_xy), sample_size, replace=False)

        n_vertical_segments = 0
        vertical_details = []

        for i in sample_idx:
            neighbors = tree_xy.query_ball_point(ground_xy[i], r=1.0)
            if len(neighbors) >= 5:
                z_vals = ground_pts_3d[neighbors, 2]
                dz = z_vals.max() - z_vals.min()
                if dz > 0.5:
                    n_vertical_segments += 1
                    if len(vertical_details) < 5:  # Guardar solo 5 ejemplos
                        vertical_details.append({
                            'pos': ground_pts_3d[i],
                            'dz': dz,
                            'n_pts': len(neighbors),
                            'z_range': (z_vals.min(), z_vals.max())
                        })

        pct_vertical = 100 * n_vertical_segments / sample_size
        print(f"\n  Segmentos con DeltaZ > 0.5m en radio 1m: {n_vertical_segments}/{sample_size} ({pct_vertical:.1f}%)")

        if vertical_details:
            print(f"\n  Ejemplos de segmentos verticales clasificados como ground:")
            for j, d in enumerate(vertical_details):
                print(f"    #{j+1}: pos=({d['pos'][0]:.1f}, {d['pos'][1]:.1f}, {d['pos'][2]:.1f}), "
                      f"DeltaZ={d['dz']:.2f}m, Z=[{d['z_range'][0]:.2f}, {d['z_range'][1]:.2f}], "
                      f"{d['n_pts']} vecinos")

# ============================================================
# Resumen final
# ============================================================
print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

# Ground bien clasificado
pw_ground_correct = np.sum(pw_ground_mask & gt_ground_mask)
pw_ground_total = np.sum(pw_ground_mask)
print(f"\nPatchwork++ ground ({pw_ground_total} puntos):")
print(f"  Correctos (GT=ground):     {pw_ground_correct:>7d} ({100*pw_ground_correct/max(1,pw_ground_total):.1f}%)")
print(f"  INCORRECTOS (GT=obstacle): {n_misclassified:>7d} ({100*n_misclassified/max(1,pw_ground_total):.1f}%)")

n_unlabeled_in_ground = np.sum(pw_ground_mask & gt_unlabeled_mask)
print(f"  Unlabeled/otro:            {n_unlabeled_in_ground:>7d} ({100*n_unlabeled_in_ground/max(1,pw_ground_total):.1f}%)")

# Conclusion
print(f"\nCONCLUSION:")
if n_misclassified > 100:
    print(f"  SI - Patchwork++ clasifica {n_misclassified} obstaculos como ground")
    print(f"  Esto representa el {100*n_misclassified/max(1,np.sum(gt_obstacle_mask)):.1f}% de los obstaculos GT")
    print(f"  Wall rejection ES NECESARIO")
else:
    print(f"  El problema es menor en esta escena: solo {n_misclassified} obstaculos mal clasificados")
    print(f"  Pero puede ser mas grave en escenas con edificios cercanos")
