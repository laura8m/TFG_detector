#!/usr/bin/env python3
"""
Debug: Analizar el problema de probabilidad binaria en Stage 3.

Investiga:
1. ¿Cuántos obstacles Stage 2 detecta per-point?
2. ¿Cuántos se proyectan al range image?
3. ¿Cuántos tienen belief > threshold en Stage 3?
4. ¿Por qué el recall cae de 91.6% a 43.1%?
"""

import numpy as np
import sys
from pathlib import Path

# Agregar paths necesarios
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


def load_kitti_scan(scan_id=0):
    """Carga scan KITTI (.bin) y labels SemanticKITTI (.label)."""
    velodyne_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne")
    labels_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels")

    scan_file = velodyne_path / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]

    label_file = labels_path / f"{scan_id:06d}.label"
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF

    return points, semantic_labels


def get_ground_truth_masks(semantic_labels):
    """Genera máscaras de ground truth."""
    obstacle_classes = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,  # Persons/cyclists
        50, 51, 52,  # Buildings/fences/walls
        60, 70, 71,  # Trunk, vegetation
        80, 81,  # Poles, traffic signs
        252, 253, 254, 255, 256, 257, 258, 259  # Other movable
    ]
    obstacle_mask = np.isin(semantic_labels, obstacle_classes)
    ground_mask = np.isin(semantic_labels, [40, 44, 48, 49, 72])

    return {
        'obstacle': obstacle_mask,
        'ground': ground_mask
    }


print("=" * 80)
print("DEBUG: BINARY PROBABILITY EN STAGE 3")
print("=" * 80)
print()

# Cargar datos
points, semantic_labels = load_kitti_scan(0)
gt_masks = get_ground_truth_masks(semantic_labels)

print(f"✓ Frame 0: {len(points)} puntos")
print(f"  Ground truth obstacles: {gt_masks['obstacle'].sum()} ({100*gt_masks['obstacle'].sum()/len(points):.1f}%)")
print()

# ========================================
# PASO 1: STAGE 2 (per-point)
# ========================================
print("-" * 80)
print("PASO 1: STAGE 2 (per-point detection)")
print("-" * 80)

config = PipelineConfig(
    enable_hybrid_wall_rejection=True,
    enable_hcd=True,
    enable_temporal_filter=False,
    verbose=False
)

pipeline = LidarPipelineSuite(config)
stage2_result = pipeline.stage2_complete(points)

# Analizar detecciones Stage 2
stage2_obs_mask = stage2_result['obs_mask']
stage2_likelihood = stage2_result['likelihood']

n_stage2_obs = stage2_obs_mask.sum()
n_stage2_obs_gt = (stage2_obs_mask & gt_masks['obstacle']).sum()

print(f"✓ Stage 2 detectó {n_stage2_obs} obstacles per-point")
print(f"  TP (GT obstacles detectados): {n_stage2_obs_gt} / {gt_masks['obstacle'].sum()} ({100*n_stage2_obs_gt/gt_masks['obstacle'].sum():.1f}%)")
print(f"  Recall Stage 2: {100*n_stage2_obs_gt/gt_masks['obstacle'].sum():.2f}%")
print()

# ========================================
# PASO 2: CONVERTIR A PROBABILIDAD BINARIA
# ========================================
print("-" * 80)
print("PASO 2: Convertir likelihood → probabilidad binaria")
print("-" * 80)

# Convertir likelihood continua a binaria (threshold > 0)
likelihood_binary = (stage2_likelihood > 0).astype(np.float32)

n_binary_obs = (likelihood_binary == 1.0).sum()
n_binary_obs_gt = ((likelihood_binary == 1.0) & gt_masks['obstacle']).sum()

print(f"✓ Probabilidad binaria: {n_binary_obs} puntos con P=1.0")
print(f"  TP (GT obstacles con P=1.0): {n_binary_obs_gt} / {gt_masks['obstacle'].sum()} ({100*n_binary_obs_gt/gt_masks['obstacle'].sum():.1f}%)")
print()

# Estadísticas likelihood continua vs binaria
print("Likelihood continua (Stage 2):")
print(f"  Min: {stage2_likelihood.min():.2f}, Max: {stage2_likelihood.max():.2f}, Mean: {stage2_likelihood.mean():.2f}")
print()
print("Probabilidad binaria (convertida):")
print(f"  Valores únicos: {np.unique(likelihood_binary)}")
print(f"  P=1.0: {(likelihood_binary==1.0).sum()} puntos ({100*(likelihood_binary==1.0).sum()/len(points):.1f}%)")
print(f"  P=0.0: {(likelihood_binary==0.0).sum()} puntos ({100*(likelihood_binary==0.0).sum()/len(points):.1f}%)")
print()

# ========================================
# PASO 3: PROYECTAR A RANGE IMAGE
# ========================================
print("-" * 80)
print("PASO 3: Proyectar probabilidad binaria → range image")
print("-" * 80)

range_proj = pipeline.project_to_range_image(
    points=points,
    likelihood=likelihood_binary,
    use_binary_probability=False  # Ya es binaria
)

likelihood_image = range_proj['likelihood_image']
range_image = range_proj['range_image']
u = range_proj['u']
v = range_proj['v']
valid_mask = range_proj['valid_mask']

# Contar píxeles ocupados
valid_pixels = range_image > 0
n_pixels_occupied = valid_pixels.sum()

# Contar píxeles con P=1.0 (obstacle)
obstacle_pixels = (likelihood_image == 1.0) & valid_pixels
n_obstacle_pixels = obstacle_pixels.sum()

print(f"✓ Range image: {range_proj['range_image'].shape}")
print(f"  Píxeles ocupados: {n_pixels_occupied}")
print(f"  Píxeles con P=1.0 (obstacle): {n_obstacle_pixels} ({100*n_obstacle_pixels/n_pixels_occupied:.1f}%)")
print()

# Analizar compresión 20:1
n_points_binary_obs = (likelihood_binary == 1.0).sum()
compression_ratio = n_points_binary_obs / n_obstacle_pixels if n_obstacle_pixels > 0 else 0

print(f"Compresión 20:1 verificación:")
print(f"  Puntos con P=1.0 (per-point): {n_points_binary_obs}")
print(f"  Píxeles con P=1.0 (range image): {n_obstacle_pixels}")
print(f"  Ratio: {compression_ratio:.1f}:1")
print()

# ========================================
# PASO 4: BAYES FILTER
# ========================================
print("-" * 80)
print("PASO 4: Bayes Filter (single frame)")
print("-" * 80)

# Reiniciar pipeline para Stage 3
pipeline_stage3 = LidarPipelineSuite(PipelineConfig(
    enable_hybrid_wall_rejection=True,
    enable_hcd=True,
    enable_temporal_filter=True,
    prob_threshold_obs=0.35,
    verbose=False
))

stage3_result = pipeline_stage3.stage3_complete(points, delta_pose=None, use_binary_probability=True)

belief_map = stage3_result['belief_map']
obs_belief_mask = stage3_result['obs_belief_mask']

# Estadísticas belief_map
print(f"✓ Belief map stats:")
print(f"  Shape: {belief_map.shape}")
print(f"  Mean: {belief_map[valid_pixels].mean():.2f}")
print(f"  Std: {belief_map[valid_pixels].std():.2f}")
print(f"  Min: {belief_map[valid_pixels].min():.2f}")
print(f"  Max: {belief_map[valid_pixels].max():.2f}")
print()

print(f"✓ Obstacle pixels (belief > threshold): {obs_belief_mask.sum()} / {valid_pixels.sum()} ({100*obs_belief_mask.sum()/valid_pixels.sum():.1f}%)")
print()

# ========================================
# PASO 5: CONVERTIR BELIEF MASK → POINT MASK
# ========================================
print("-" * 80)
print("PASO 5: Convertir belief_mask (H×W) → point_mask (N)")
print("-" * 80)

# Crear máscara per-point basada en belief_map
obs_point_mask_stage3 = np.zeros(len(points), dtype=bool)

for i in range(len(points)):
    if valid_mask[i]:
        ui = u[i]
        vi = v[i]
        if obs_belief_mask[ui, vi]:
            obs_point_mask_stage3[i] = True

n_stage3_obs = obs_point_mask_stage3.sum()
n_stage3_obs_gt = (obs_point_mask_stage3 & gt_masks['obstacle']).sum()

print(f"✓ Stage 3 detectó {n_stage3_obs} obstacles (per-point)")
print(f"  TP (GT obstacles detectados): {n_stage3_obs_gt} / {gt_masks['obstacle'].sum()} ({100*n_stage3_obs_gt/gt_masks['obstacle'].sum():.1f}%)")
print(f"  Recall Stage 3: {100*n_stage3_obs_gt/gt_masks['obstacle'].sum():.2f}%")
print()

# ========================================
# ANÁLISIS: ¿DÓNDE SE PIERDEN LOS GT OBSTACLES?
# ========================================
print("=" * 80)
print("ANÁLISIS: ¿DÓNDE SE PIERDEN LOS GT OBSTACLES?")
print("=" * 80)
print()

# GT obstacles perdidos en cada paso
gt_obstacles = gt_masks['obstacle']
n_gt_total = gt_obstacles.sum()

# Paso 1 → 2: Stage 2 detecta
lost_stage2 = gt_obstacles & (~stage2_obs_mask)
n_lost_stage2 = lost_stage2.sum()

# Paso 2 → 3: Conversión binaria
binary_obs_mask = likelihood_binary == 1.0
lost_binary = gt_obstacles & stage2_obs_mask & (~binary_obs_mask)
n_lost_binary = lost_binary.sum()

# Paso 3 → 4: Proyección range image (compresión 20:1)
# Crear máscara de GT que se proyectó correctamente
gt_in_range_image = np.zeros(len(points), dtype=bool)
for i in np.where(gt_obstacles)[0]:
    if valid_mask[i]:
        ui = u[i]
        vi = v[i]
        if likelihood_image[ui, vi] == 1.0:
            gt_in_range_image[i] = True

lost_projection = gt_obstacles & binary_obs_mask & (~gt_in_range_image)
n_lost_projection = lost_projection.sum()

# Paso 4 → 5: Bayes Filter threshold
lost_bayes = gt_obstacles & gt_in_range_image & (~obs_point_mask_stage3)
n_lost_bayes = lost_bayes.sum()

print(f"GT obstacles totales: {n_gt_total}")
print()
print(f"Perdidos en Stage 2 (delta_r): {n_lost_stage2} ({100*n_lost_stage2/n_gt_total:.1f}%)")
print(f"Perdidos en conversión binaria: {n_lost_binary} ({100*n_lost_binary/n_gt_total:.1f}%)")
print(f"Perdidos en proyección range image: {n_lost_projection} ({100*n_lost_projection/n_gt_total:.1f}%)")
print(f"Perdidos en Bayes Filter threshold: {n_lost_bayes} ({100*n_lost_bayes/n_gt_total:.1f}%)")
print()
print(f"Detectados finalmente (Stage 3): {n_stage3_obs_gt} ({100*n_stage3_obs_gt/n_gt_total:.1f}%)")
print()

# Verificación
total_accounted = n_stage3_obs_gt + n_lost_stage2 + n_lost_binary + n_lost_projection + n_lost_bayes
print(f"Verificación: {total_accounted} / {n_gt_total} ({100*total_accounted/n_gt_total:.1f}%)")

print()
print("=" * 80)
print("DEBUG COMPLETADO")
print("=" * 80)
