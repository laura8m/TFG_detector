#!/usr/bin/env python3
"""
Test: Análisis por clase y por distancia de PW++ vanilla vs PW++ + WR.

1. Por clase semántica: F1/Recall por tipo de obstáculo (coches, peatones, etc.)
2. Por distancia: F1 en rangos (0-20m, 20-40m, 40-80m)

Evaluado en SemanticKITTI val (seq 08).
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file

# ========================================
# ETIQUETAS SemanticKITTI
# ========================================

# Clases de obstáculo con nombres
OBSTACLE_CLASSES = {
    'car':        [10, 252],
    'bicycle':    [11, 253],
    'motorcycle': [15, 254],
    'truck':      [16, 258],
    'other-veh':  [18, 259],
    'person':     [30, 254],
    'bicyclist':  [31, 255],
    'motorcycl.': [32, 256],
    'bus':        [13, 257],
    'building':   [50],
    'fence':      [51],
    'vegetation': [70],
    'trunk':      [71],
    'pole':       [80],
    'sign':       [81],
    'other-obj':  [20],
}

OBSTACLE_LABELS_ALL = set()
for labels in OBSTACLE_CLASSES.values():
    OBSTACLE_LABELS_ALL.update(labels)

IGNORE_LABELS = {0, 1, 52, 99}

# Rangos de distancia (metros)
DISTANCE_RANGES = [
    (0, 10, '0-10m'),
    (10, 20, '10-20m'),
    (20, 30, '20-30m'),
    (30, 40, '30-40m'),
    (40, 60, '40-60m'),
    (60, 80, '60-80m'),
]

# ========================================
# CONFIGURACIONES
# ========================================

CONFIGS = {
    'PW++ vanilla': PipelineConfig(
        enable_hybrid_wall_rejection=False,
        enable_delta_r=False,
        verbose=False,
    ),
    'PW++ + WR': PipelineConfig(
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=0.9,
        wall_height_diff_threshold=0.2,
        wall_kdtree_radius=0.3,
        enable_delta_r=False,
        verbose=False,
    ),
}


def load_scan(scan_id, seq):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    return points, semantic_labels


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Análisis por clase y distancia')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--seq', type=str, default='08')
    args = parser.parse_args()

    info = get_sequence_info(args.seq)
    velodyne_dir = info['data_dir'] / 'velodyne'
    scan_files = sorted(velodyne_dir.glob('*.bin'))
    scan_ids = [int(f.stem) for f in scan_files][::args.stride]

    print("=" * 120)
    print(f"ANÁLISIS POR CLASE Y DISTANCIA — Seq {args.seq} | {len(scan_ids)} frames | stride={args.stride}")
    print("=" * 120)

    # Acumuladores por clase
    class_accum = {}
    for cls_name in OBSTACLE_CLASSES:
        class_accum[cls_name] = {cfg: {'tp': 0, 'fp': 0, 'fn': 0} for cfg in CONFIGS}

    # Acumuladores por distancia
    dist_accum = {}
    for _, _, range_name in DISTANCE_RANGES:
        dist_accum[range_name] = {cfg: {'tp': 0, 'fp': 0, 'fn': 0} for cfg in CONFIGS}

    # Acumuladores globales
    global_accum = {cfg: {'tp': 0, 'fp': 0, 'fn': 0} for cfg in CONFIGS}

    for i, scan_id in enumerate(scan_ids):
        pts, labels = load_scan(scan_id, args.seq)
        valid_mask = np.ones(len(pts), dtype=bool)
        for lbl in IGNORE_LABELS:
            valid_mask &= (labels != lbl)

        # Distancia de cada punto al sensor
        distances = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)

        gt_obs = np.zeros(len(pts), dtype=bool)
        for lbl in OBSTACLE_LABELS_ALL:
            gt_obs |= (labels == lbl)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{len(scan_ids)} (scan {scan_id})")

        for cfg_name, config in CONFIGS.items():
            pipeline = LidarPipelineSuite(config)
            s1 = pipeline.stage1_complete(pts)
            pred_obs = np.zeros(len(pts), dtype=bool)
            pred_obs[s1['nonground_indices']] = True

            # --- Métricas globales ---
            gt_v = gt_obs & valid_mask
            pred_v = pred_obs & valid_mask
            global_accum[cfg_name]['tp'] += int(np.sum(gt_v & pred_v))
            global_accum[cfg_name]['fp'] += int(np.sum((~gt_v) & pred_v))
            global_accum[cfg_name]['fn'] += int(np.sum(gt_v & (~pred_v)))

            # --- Métricas por clase ---
            for cls_name, cls_labels in OBSTACLE_CLASSES.items():
                cls_mask = np.zeros(len(pts), dtype=bool)
                for lbl in cls_labels:
                    cls_mask |= (labels == lbl)
                cls_valid = cls_mask & valid_mask

                # TP: puntos de esta clase detectados como obstáculo
                tp = int(np.sum(cls_valid & pred_obs))
                # FN: puntos de esta clase NO detectados
                fn = int(np.sum(cls_valid & (~pred_obs)))
                # FP no tiene sentido por clase individual (no sabemos qué clase predice)

                class_accum[cls_name][cfg_name]['tp'] += tp
                class_accum[cls_name][cfg_name]['fn'] += fn

            # --- Métricas por distancia ---
            for d_min, d_max, range_name in DISTANCE_RANGES:
                d_mask = (distances >= d_min) & (distances < d_max)
                gt_d = gt_obs & valid_mask & d_mask
                pred_d = pred_obs & valid_mask & d_mask

                tp = int(np.sum(gt_d & pred_d))
                fp = int(np.sum((~gt_d) & pred_d & d_mask & valid_mask))
                fn = int(np.sum(gt_d & (~pred_d)))

                dist_accum[range_name][cfg_name]['tp'] += tp
                dist_accum[range_name][cfg_name]['fp'] += fp
                dist_accum[range_name][cfg_name]['fn'] += fn

    # ========================================
    # RESULTADOS GLOBALES
    # ========================================
    print(f"\n{'='*120}")
    print("RESULTADOS GLOBALES")
    print(f"{'='*120}")
    print(f"  {'Config':<25} {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8}")
    print(f"  {'-'*60}")
    for cfg_name in CONFIGS:
        g = global_accum[cfg_name]
        p, r, f1 = compute_f1(g['tp'], g['fp'], g['fn'])
        iou = g['tp'] / (g['tp'] + g['fp'] + g['fn']) if (g['tp'] + g['fp'] + g['fn']) > 0 else 0
        print(f"  {cfg_name:<25} {100*f1:>7.2f}% {100*iou:>7.2f}% {100*p:>7.2f}% {100*r:>7.2f}%")

    # ========================================
    # RESULTADOS POR CLASE (Recall)
    # ========================================
    print(f"\n{'='*120}")
    print("ANÁLISIS POR CLASE — Recall (% de puntos de cada clase detectados como obstáculo)")
    print(f"{'='*120}")

    # Header
    cfg_names = list(CONFIGS.keys())
    header = f"  {'Clase':<15} {'N puntos':>10}"
    for cfg in cfg_names:
        header += f" {cfg:>18}"
    header += f" {'Delta':>10}"
    print(header)
    print(f"  {'-'*90}")

    # Ordenar por número de puntos (más frecuentes primero)
    class_order = sorted(OBSTACLE_CLASSES.keys(),
                         key=lambda c: class_accum[c][cfg_names[0]]['tp'] + class_accum[c][cfg_names[0]]['fn'],
                         reverse=True)

    for cls_name in class_order:
        total_pts = class_accum[cls_name][cfg_names[0]]['tp'] + class_accum[cls_name][cfg_names[0]]['fn']
        if total_pts == 0:
            continue

        recalls = []
        row = f"  {cls_name:<15} {total_pts:>10}"
        for cfg in cfg_names:
            tp = class_accum[cls_name][cfg]['tp']
            fn = class_accum[cls_name][cfg]['fn']
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            row += f" {100*recall:>17.2f}%"

        delta = recalls[1] - recalls[0]
        row += f" {100*delta:>+9.2f}%"
        print(row)

    # ========================================
    # RESULTADOS POR DISTANCIA
    # ========================================
    print(f"\n{'='*120}")
    print("ANÁLISIS POR DISTANCIA — F1 por rango")
    print(f"{'='*120}")

    header = f"  {'Rango':<10} {'N obs pts':>10}"
    for cfg in cfg_names:
        header += f" {'F1':>8} {'P':>8} {'R':>8}"
    header += f" {'Delta F1':>10}"
    print(header)
    print(f"  {'-'*100}")

    for d_min, d_max, range_name in DISTANCE_RANGES:
        total_obs = dist_accum[range_name][cfg_names[0]]['tp'] + dist_accum[range_name][cfg_names[0]]['fn']
        row = f"  {range_name:<10} {total_obs:>10}"

        f1s = []
        for cfg in cfg_names:
            d = dist_accum[range_name][cfg]
            p, r, f1 = compute_f1(d['tp'], d['fp'], d['fn'])
            f1s.append(f1)
            row += f" {100*f1:>7.2f}% {100*p:>7.2f}% {100*r:>7.2f}%"

        delta = f1s[1] - f1s[0]
        row += f" {100*delta:>+9.2f}%"
        print(row)

    print(f"\n  Total frames: {len(scan_ids)}")


if __name__ == '__main__':
    main()
