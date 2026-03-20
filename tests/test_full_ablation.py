#!/usr/bin/env python3
"""
Test Ablation Completo del Pipeline

Ablation acumulativo: cada stage se añade progresivamente para medir su contribución.
- Config 1: Stage 2 solo (single-frame baseline)
- Config 2: Stage 2 → 3 (+ DBSCAN Cluster Filtering) [pipeline completo]

Métricas: Precision, Recall, F1, FP, FN + timing desglosado por stage.
Secuencias: KITTI 00 (urbano) y 04 (highway).
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
# UTILIDADES
# ========================================

def load_kitti_scan(scan_id: int, seq: str):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    return points, semantic_labels


def compute_metrics(gt_mask, pred_mask, valid_mask=None):
    if valid_mask is not None:
        gt_mask = gt_mask & valid_mask
        pred_mask = pred_mask & valid_mask
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


IGNORE_LABELS = [0, 1, 52, 99]  # learning_map → 0 en SemanticKITTI


def get_gt_obstacle_mask(semantic_labels):
    """SemanticKITTI obstacle labels (NO 72=terrain, NO 52/99=ignored, SI 252-259=moving)"""
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,                    # Persons
        50, 51,                        # Structures (NO 52=other-structure)
        70, 71,                        # Vegetation (NO 72=terrain)
        80, 81,                        # Poles/signs
        252, 253, 254, 255, 256, 257, 258, 259  # Moving objects
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


def get_valid_mask(semantic_labels):
    """Máscara de puntos válidos (excluye unlabeled, outlier, other-structure, other-object)"""
    mask = np.ones(len(semantic_labels), dtype=bool)
    for label in IGNORE_LABELS:
        mask &= (semantic_labels != label)
    return mask


# ========================================
# ABLATION CONFIGS
# ========================================

def get_ablation_configs():
    """Retorna dict de configs para ablation acumulativo."""
    configs = {
        'Stage 2 (baseline)': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_cluster_filtering=False,
            verbose=False
        ),
        'Stage 2→3': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_cluster_filtering=True,
            cluster_min_pts=15,
            verbose=False
        ),
    }

    return configs


# ========================================
# RUNNER POR CONFIG
# ========================================

def run_config(config_name, config, seq, scan_start, n_frames, poses):
    """Ejecutar una config y devolver métricas + timings del último frame"""

    pipeline = LidarPipelineSuite(config)

    # Stage 2 solo: single-frame (solo el último frame)
    if config_name == 'Stage 2 (baseline)':
        scan_ref = scan_start + n_frames - 1
        pts, _ = load_kitti_scan(scan_ref, seq)
        result = pipeline.stage2_complete(pts)
        return result, {
            'timing_s12_ms': result.get('timing_total_ms', 0),
            'timing_stage3_ms': 0,
            'timing_total_ms': result.get('timing_total_ms', 0),
        }

    # Stage 2→3: Stage 2 + DBSCAN cluster filtering
    if config_name in ('Stage 2→3', 'Stage 2→3 (sin HCD)'):
        scan_ref = scan_start + n_frames - 1
        pts, _ = load_kitti_scan(scan_ref, seq)
        result = pipeline.stage3_complete(pts)
        t_total = result.get('timing_total_ms', 0)
        t_s3 = result.get('timing_stage3_ms', 0)
        t_s12 = t_total - t_s3
        return result, {
            'timing_s12_ms': max(0, t_s12),
            'timing_stage3_ms': t_s3,
            'timing_total_ms': t_total,
        }

    # Fallback: Stage 2 sin HCD (single-frame)
    scan_ref = scan_start + n_frames - 1
    pts, _ = load_kitti_scan(scan_ref, seq)
    result = pipeline.stage2_complete(pts)
    return result, {
        'timing_s12_ms': result.get('timing_total_ms', 0),
        'timing_stage3_ms': 0,
        'timing_total_ms': result.get('timing_total_ms', 0),
    }


# ========================================
# TEST POR SECUENCIA
# ========================================

def test_sequence(seq, scan_start, n_frames):
    info = get_sequence_info(seq)
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])

    # GT del último frame
    scan_ref = scan_start + n_frames - 1
    _, labels_ref = load_kitti_scan(scan_ref, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)
    valid_mask = get_valid_mask(labels_ref)

    print("=" * 90)
    print(f"SECUENCIA {seq} | Frames {scan_start}-{scan_ref} ({n_frames} frames) | GT obstacles: {gt_mask.sum()}")
    print("=" * 90)

    configs = get_ablation_configs()
    all_metrics = {}
    all_timings = {}

    for name, config in configs.items():
        print(f"\n  Ejecutando: {name}...", end=" ", flush=True)
        t0 = time.time()

        result, timings = run_config(name, config, seq, scan_start, n_frames, poses)
        metrics = compute_metrics(gt_mask, result['obs_mask'], valid_mask)

        all_metrics[name] = metrics
        all_timings[name] = timings

        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s)")

    # ========================================
    # TABLA DE MÉTRICAS
    # ========================================
    print(f"\n{'='*90}")
    print(f"MÉTRICAS - SECUENCIA {seq}")
    print(f"{'='*90}")
    print(f"\n{'Config':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'FP':>8} {'FN':>8}")
    print("-" * 94)
    for name, m in all_metrics.items():
        print(f"{name:<28} {100*m['precision']:>9.1f}% {100*m['recall']:>9.1f}% {100*m['f1']:>9.1f}% {100*m['iou']:>9.1f}% {m['fp']:>8} {m['fn']:>8}")

    # ========================================
    # TABLA DE TIMING
    # ========================================
    print(f"\n{'='*90}")
    print(f"TIMING (último frame) - SECUENCIA {seq}")
    print(f"{'='*90}")
    print(f"\n{'Config':<28} {'S1+S2':>8} {'S3':>8} {'Total':>8}")
    print("-" * 60)
    for name, t in all_timings.items():
        print(f"{name:<28} {t['timing_s12_ms']:>7.0f}ms {t['timing_stage3_ms']:>7.0f}ms {t['timing_total_ms']:>7.0f}ms")

    # ========================================
    # CONTRIBUCIÓN DE CADA STAGE
    # ========================================
    print(f"\n{'='*90}")
    print(f"CONTRIBUCIÓN POR STAGE - SECUENCIA {seq}")
    print(f"{'='*90}")

    # Stage 3 contribution: (2→3) vs (2)
    if 'Stage 2 (baseline)' in all_metrics and 'Stage 2→3' in all_metrics:
        m_base = all_metrics['Stage 2 (baseline)']
        m_s3 = all_metrics['Stage 2→3']
        print(f"\n  Stage 3 (DBSCAN Cluster Filtering):")
        print(f"    F1:        {100*(m_s3['f1']-m_base['f1']):+.2f}%")
        print(f"    IoU:       {100*(m_s3['iou']-m_base['iou']):+.2f}%")
        print(f"    Precision: {100*(m_s3['precision']-m_base['precision']):+.2f}%")
        print(f"    Recall:    {100*(m_s3['recall']-m_base['recall']):+.2f}%")
        print(f"    FP:        {m_s3['fp']-m_base['fp']:+d}")

    # Pipeline completo vs baseline
    if 'Stage 2 (baseline)' in all_metrics and 'Stage 2→3' in all_metrics:
        m_base = all_metrics['Stage 2 (baseline)']
        m_full = all_metrics['Stage 2→3']
        print(f"\n  TOTAL (Pipeline completo vs Stage 2 solo):")
        print(f"    F1:        {100*(m_full['f1']-m_base['f1']):+.2f}%")
        print(f"    IoU:       {100*(m_full['iou']-m_base['iou']):+.2f}%")
        print(f"    Precision: {100*(m_full['precision']-m_base['precision']):+.2f}%")
        print(f"    Recall:    {100*(m_full['recall']-m_base['recall']):+.2f}%")

    return all_metrics, all_timings


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Ablation Study')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    args = parser.parse_args()

    all_results = {}

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', args.scan_start, args.n_frames)

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', args.scan_start, args.n_frames)

    # Resumen global
    if len(all_results) > 1 and all(v is not None for v in all_results.values()):
        print("\n" + "=" * 90)
        print("RESUMEN GLOBAL (ambas secuencias)")
        print("=" * 90)

        config_names = list(all_results['04'][0].keys())
        print(f"\n{'Config':<28} {'Seq 04 F1':>10} {'Seq 00 F1':>10} {'Media F1':>10} {'Seq 04 IoU':>11} {'Seq 00 IoU':>11} {'Media IoU':>10}")
        print("-" * 100)
        for name in config_names:
            f1_04 = all_results['04'][0][name]['f1']
            f1_00 = all_results['00'][0][name]['f1']
            f1_avg = (f1_04 + f1_00) / 2
            iou_04 = all_results['04'][0][name]['iou']
            iou_00 = all_results['00'][0][name]['iou']
            iou_avg = (iou_04 + iou_00) / 2
            print(f"{name:<28} {100*f1_04:>9.1f}% {100*f1_00:>9.1f}% {100*f1_avg:>9.1f}% {100*iou_04:>10.1f}% {100*iou_00:>10.1f}% {100*iou_avg:>9.1f}%")


if __name__ == '__main__':
    main()
