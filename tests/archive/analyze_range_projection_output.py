#!/usr/bin/env python3
"""
Analiza la salida de range_projection.py (belief_prob.npy) y calcula métricas.

Lee:
- belief_prob_scan_N.npy: (H, W) probabilidad final del Bayes Filter
- Carga puntos KITTI y GT labels manualmente
- Proyecta puntos a range image
- Compara belief_prob con GT

Uso:
    python3 analyze_range_projection_output.py --scan 0
"""

import numpy as np
from pathlib import Path
import argparse

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


def project_points_to_range_image(points, H=64, W=2048):
    """
    Proyecta puntos 3D a range image (igual que range_projection.py).

    Returns:
        u: (N,) índices de fila
        v: (N,) índices de columna
        valid_mask: (N,) máscara de puntos válidos
    """
    # Parámetros LiDAR (Velodyne HDL-64E)
    fov_up = 3.0
    fov_down = -25.0
    fov_range = fov_up - fov_down

    # Calcular rango y ángulos
    r = np.linalg.norm(points, axis=1)
    pitch = np.arcsin(np.clip(points[:, 2] / r, -1.0, 1.0))  # Vertical angle
    yaw = np.arctan2(points[:, 1], points[:, 0])  # Horizontal angle

    # Convertir a píxeles
    pitch_deg = np.degrees(pitch)
    proj_y = (pitch_deg - fov_down) / fov_range  # 0..1
    proj_y = 1.0 - proj_y  # Invertir (top = 0)

    proj_x = 0.5 * (yaw / np.pi + 1.0)  # 0..1

    u = np.floor(proj_y * H).astype(np.int32)
    v = np.floor(proj_x * W).astype(np.int32)

    # Clamp
    u = np.clip(u, 0, H - 1)
    v = np.clip(v, 0, W - 1)

    # Máscara válida (mismo criterio que range_projection.py)
    min_range = 2.7
    max_range = 80.0
    valid_mask = (r > min_range) & (r < max_range)

    return u, v, valid_mask


def compute_detection_metrics(gt_mask, pred_mask):
    """Calcula Precision, Recall, F1."""
    TP = np.sum(gt_mask & pred_mask)
    FP = np.sum((~gt_mask) & pred_mask)
    FN = np.sum(gt_mask & (~pred_mask))
    TN = np.sum((~gt_mask) & (~pred_mask))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("ANÁLISIS DE RANGE_PROJECTION.PY OUTPUT")
    print("=" * 80)
    print(f"Scan: {args.scan}")
    print()

    # Cargar belief_prob de range_projection
    belief_file = Path(f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/tests/range_projection_output/belief_prob_scan_{args.scan}.npy")

    if not belief_file.exists():
        print(f"[ERROR] No se encontró {belief_file}")
        print("[INFO] Ejecuta primero:")
        print(f"  bash launch_range_projection.sh {args.scan} {args.scan}")
        return

    belief_prob = np.load(belief_file)
    print(f"✓ Cargado belief_prob: shape={belief_prob.shape}")
    print(f"  Stats: min={belief_prob.min():.3f}, max={belief_prob.max():.3f}, mean={belief_prob.mean():.3f}")
    print()

    # Cargar datos KITTI
    points, semantic_labels = load_kitti_scan(args.scan)
    gt_masks = get_ground_truth_masks(semantic_labels)

    print(f"✓ Cargado scan {args.scan}: {len(points)} puntos")
    print(f"  GT obstacles: {gt_masks['obstacle'].sum()} ({100*gt_masks['obstacle'].sum()/len(points):.1f}%)")
    print()

    # Proyectar puntos a range image
    u, v, valid_mask = project_points_to_range_image(points)

    print(f"✓ Proyección: {valid_mask.sum()} puntos válidos / {len(points)} totales")
    print()

    # Crear máscara de detección per-point desde belief_prob
    # Threshold: P > 0.5 → obstacle
    threshold_prob = 0.5
    belief_mask_2d = belief_prob > threshold_prob  # (H, W)

    # Proyectar a per-point mask
    obstacle_mask = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        if valid_mask[i]:
            obstacle_mask[i] = belief_mask_2d[u[i], v[i]]

    print(f"✓ Máscara de detección creada")
    print(f"  Obstacles detectados: {obstacle_mask.sum()} ({100*obstacle_mask.sum()/len(points):.1f}%)")
    print()

    # Calcular métricas
    metrics = compute_detection_metrics(gt_masks['obstacle'], obstacle_mask)

    print("=" * 80)
    print("MÉTRICAS DE DETECCIÓN (range_projection.py)")
    print("=" * 80)
    print()
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    print()
    print(f"  TP: {metrics['TP']}")
    print(f"  FP: {metrics['FP']}")
    print(f"  FN: {metrics['FN']}")
    print(f"  TN: {metrics['TN']}")
    print()

    print("=" * 80)
    print("COMPARACIÓN CON LIDAR_PIPELINE_SUITE.PY")
    print("=" * 80)
    print()
    print("lidar_pipeline_suite.py (Stage 2):")
    print("  Recall: 91.60%, Precision: 63.48%, F1: 74.99%")
    print()
    print("lidar_pipeline_suite.py (Stage 3 - CLOSEST WINS):")
    print("  Recall: 43.09%, Precision: 67.81%, F1: 52.70%")
    print()
    print(f"range_projection.py (Scan {args.scan}):")
    print(f"  Recall: {metrics['recall']*100:.2f}%, Precision: {metrics['precision']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")
    print()

    if metrics['recall'] < 0.5:
        print("✗ RECALL BAJO (<50%): Confirma el problema de compresión 20:1")
        print("  → range_projection.py usa CLOSEST WINS como lidar_pipeline_suite.py")
    elif metrics['recall'] > 0.9:
        print("✓ RECALL ALTO (>90%): range_projection.py funciona bien")
        print("  → Necesitas investigar qué hace diferente vs lidar_pipeline_suite.py")
    else:
        print("⚠ RECALL MODERADO (50-90%): Resultado intermedio")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
