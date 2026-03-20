#!/usr/bin/env python3
"""
Test: Ablation de Stage 1 — Patchwork++ vanilla vs Stage 1 completo.

Compara 3 configuraciones de segmentación de suelo:
1. Patchwork++ vanilla (sin wall rejection, sin HCD)
2. Patchwork++ + Wall Rejection (sin HCD)
3. Stage 1 completo (Patchwork++ + Wall Rejection + HCD)

Métricas de ground segmentation:
- Ground Precision: de los puntos clasificados como ground, cuántos son ground real
- Ground Recall: de los puntos ground reales, cuántos se detectan
- Obstacle Leak: puntos obstáculo clasificados como ground (FP de ground = peligroso)

Métricas de detección de obstáculos (después de Stage 2):
- Obstacle Precision, Recall, F1, IoU

Timing desglosado por sub-etapa.

Evaluado en N frames de seq 00 (urbano) y seq 04 (highway).
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file

# ========================================
# DATOS
# ========================================

# SemanticKITTI ground labels
GROUND_LABELS = {40, 44, 48, 49, 60, 72}  # road, parking, sidewalk, other-ground, lane-marking, terrain

# SemanticKITTI obstacle labels
OBSTACLE_LABELS = {
    10, 11, 13, 15, 16, 18, 20,       # vehicles
    30, 31, 32,                         # persons
    50, 51, 52,                         # structures
    70, 71,                             # vegetation
    80, 81,                             # poles
    99,                                 # other-object
    252, 253, 254, 255, 256, 257, 258, 259  # moving objects
}


def load_kitti_scan(scan_id, seq):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    return points, semantic_labels


def get_gt_ground_mask(semantic_labels):
    """Máscara de puntos que son ground real."""
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in GROUND_LABELS:
        mask |= (semantic_labels == label)
    return mask


def get_gt_obstacle_mask(semantic_labels):
    """Máscara de puntos que son obstáculo real."""
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in OBSTACLE_LABELS:
        mask |= (semantic_labels == label)
    return mask


def compute_metrics(gt_mask, pred_mask):
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn}


# ========================================
# CONFIGURACIONES
# ========================================

CONFIGS = {
    'Patchwork++ vanilla': PipelineConfig(
        enable_hybrid_wall_rejection=False,
        verbose=False,
    ),
    'PW++ + Wall Rejection': PipelineConfig(
        enable_hybrid_wall_rejection=True,
        verbose=False,
    ),
}


# ========================================
# TEST
# ========================================

def test_sequence(seq, n_frames, scan_start):
    print("\n" + "=" * 100)
    print(f"SECUENCIA {seq} | Frames {scan_start}-{scan_start + n_frames - 1}")
    print("=" * 100)

    # Acumuladores por configuración
    # Ground segmentation metrics
    gnd_accum = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in CONFIGS}
    # Obstacle leak: obstáculos GT clasificados como ground
    obs_leak_accum = {name: {'leaked': 0, 'total_obs': 0} for name in CONFIGS}
    # Obstacle detection metrics (después de Stage 2)
    obs_accum = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in CONFIGS}
    # Timing
    timing_s1 = {name: [] for name in CONFIGS}
    timing_s2 = {name: [] for name in CONFIGS}

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, labels = load_kitti_scan(scan_id, seq)
        gt_ground = get_gt_ground_mask(labels)
        gt_obs = get_gt_obstacle_mask(labels)

        print(f"\n  Frame {scan_id}: {len(pts)} pts | GT ground={gt_ground.sum()} obs={gt_obs.sum()}")

        for name, config in CONFIGS.items():
            pipeline = LidarPipelineSuite(config)

            # --- Stage 1 ---
            t1 = time.time()
            r1 = pipeline.stage1_complete(pts)
            t1_end = time.time()
            timing_s1[name].append((t1_end - t1) * 1000)

            # Máscara de ground predicha por Stage 1
            pred_ground = np.zeros(len(pts), dtype=bool)
            pred_ground[r1['ground_indices']] = True

            # Métricas ground segmentation
            m_gnd = compute_metrics(gt_ground, pred_ground)
            gnd_accum[name]['tp'] += m_gnd['tp']
            gnd_accum[name]['fp'] += m_gnd['fp']
            gnd_accum[name]['fn'] += m_gnd['fn']

            # Obstacle leak: obstáculos GT que Stage 1 clasifica como ground
            obs_in_ground = int(np.sum(gt_obs & pred_ground))
            obs_leak_accum[name]['leaked'] += obs_in_ground
            obs_leak_accum[name]['total_obs'] += int(gt_obs.sum())

            # --- Stage 2 (delta-r) ---
            t2 = time.time()
            r2 = pipeline.stage2_complete(pts)
            t2_end = time.time()
            # Stage 2 incluye Stage 1 internamente, así que el tiempo real de Stage 2 solo es:
            timing_s2[name].append((t2_end - t2) * 1000)

            # Métricas de detección de obstáculos
            m_obs = compute_metrics(gt_obs, r2['obs_mask'])
            obs_accum[name]['tp'] += m_obs['tp']
            obs_accum[name]['fp'] += m_obs['fp']
            obs_accum[name]['fn'] += m_obs['fn']

            n_walls = len(r1.get('rejected_walls', []))
            print(f"    {name:<30} | gnd P={100*m_gnd['precision']:.1f}% R={100*m_gnd['recall']:.1f}% | "
                  f"leak={obs_in_ground} | obs F1={100*m_obs['f1']:.1f}% | walls={n_walls} | "
                  f"S1={timing_s1[name][-1]:.0f}ms S2={timing_s2[name][-1]:.0f}ms")

    # ========================================
    # RESUMEN
    # ========================================
    print(f"\n{'='*100}")
    print(f"RESUMEN SECUENCIA {seq} ({n_frames} frames)")
    print(f"{'='*100}")

    # --- Ground Segmentation ---
    print(f"\n  GROUND SEGMENTATION (Stage 1)")
    print(f"  {'Config':<30} {'Gnd P':>8} {'Gnd R':>8} {'Gnd F1':>8} {'Obs Leak':>10} {'Leak %':>8}")
    print(f"  {'-'*80}")

    for name in CONFIGS:
        g = gnd_accum[name]
        p = g['tp'] / (g['tp'] + g['fp']) if (g['tp'] + g['fp']) > 0 else 0
        r = g['tp'] / (g['tp'] + g['fn']) if (g['tp'] + g['fn']) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        leak = obs_leak_accum[name]
        leak_pct = 100 * leak['leaked'] / leak['total_obs'] if leak['total_obs'] > 0 else 0
        print(f"  {name:<30} {100*p:>7.1f}% {100*r:>7.1f}% {100*f1:>7.1f}% {leak['leaked']:>10} {leak_pct:>7.1f}%")

    # --- Obstacle Detection ---
    print(f"\n  DETECCIÓN DE OBSTÁCULOS (Stage 2)")
    print(f"  {'Config':<30} {'Obs P':>8} {'Obs R':>8} {'Obs F1':>8} {'Obs IoU':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*80}")

    for name in CONFIGS:
        o = obs_accum[name]
        p = o['tp'] / (o['tp'] + o['fp']) if (o['tp'] + o['fp']) > 0 else 0
        r = o['tp'] / (o['tp'] + o['fn']) if (o['tp'] + o['fn']) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        iou = o['tp'] / (o['tp'] + o['fp'] + o['fn']) if (o['tp'] + o['fp'] + o['fn']) > 0 else 0
        print(f"  {name:<30} {100*p:>7.1f}% {100*r:>7.1f}% {100*f1:>7.1f}% {100*iou:>7.1f}% {o['fp']:>8} {o['fn']:>8}")

    # --- Timing ---
    print(f"\n  TIMING (media por frame)")
    print(f"  {'Config':<30} {'Stage 1':>10} {'Stage 2 total':>14}")
    print(f"  {'-'*60}")

    for name in CONFIGS:
        s1_mean = np.mean(timing_s1[name])
        s2_mean = np.mean(timing_s2[name])
        print(f"  {name:<30} {s1_mean:>9.1f}ms {s2_mean:>13.1f}ms")

    return {
        'ground': gnd_accum,
        'obs_leak': obs_leak_accum,
        'obs': obs_accum,
        'timing_s1': timing_s1,
        'timing_s2': timing_s2,
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 1 ablation: Patchwork++ vanilla vs completo')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--scan_start', type=int, default=0)
    args = parser.parse_args()

    seqs = ['04', '00'] if args.seq == 'both' else [args.seq]
    all_results = {}

    for seq in seqs:
        all_results[seq] = test_sequence(seq, args.n_frames, args.scan_start)

    # Resumen global si hay ambas secuencias
    if len(seqs) > 1:
        print("\n" + "=" * 100)
        print("RESUMEN GLOBAL (ambas secuencias)")
        print("=" * 100)

        print(f"\n  {'Config':<30} {'Mean Gnd F1':>12} {'Mean Obs F1':>12} {'Mean Leak%':>12} {'Mean S1 ms':>12}")
        print(f"  {'-'*80}")

        for name in CONFIGS:
            gnd_f1s = []
            obs_f1s = []
            leak_pcts = []
            s1_means = []

            for seq in seqs:
                res = all_results[seq]
                g = res['ground'][name]
                p = g['tp'] / (g['tp'] + g['fp']) if (g['tp'] + g['fp']) > 0 else 0
                r = g['tp'] / (g['tp'] + g['fn']) if (g['tp'] + g['fn']) > 0 else 0
                gnd_f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

                o = res['obs'][name]
                p = o['tp'] / (o['tp'] + o['fp']) if (o['tp'] + o['fp']) > 0 else 0
                r = o['tp'] / (o['tp'] + o['fn']) if (o['tp'] + o['fn']) > 0 else 0
                obs_f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

                lk = res['obs_leak'][name]
                leak_pcts.append(100 * lk['leaked'] / lk['total_obs'] if lk['total_obs'] > 0 else 0)

                s1_means.append(np.mean(res['timing_s1'][name]))

            print(f"  {name:<30} {100*np.mean(gnd_f1s):>11.1f}% {100*np.mean(obs_f1s):>11.1f}% "
                  f"{np.mean(leak_pcts):>11.1f}% {np.mean(s1_means):>11.1f}ms")


if __name__ == '__main__':
    main()
