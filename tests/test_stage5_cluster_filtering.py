#!/usr/bin/env python3
"""
Test Stage 3: DBSCAN Cluster Filtering

Compara en ambas secuencias KITTI (00 y 04):
- Stage 2 solo (baseline single-frame)
- Stage 3 = Stage 2 + DBSCAN cluster filtering

Objetivo: Validar que Stage 3 reduce FP (puntos dispersos) manteniendo recall.
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file

# ========================================
# UTILIDADES
# ========================================

def load_kitti_scan(scan_id: int, seq: str = '04'):
    """Cargar scan de KITTI con labels"""
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]

    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)

    return points, semantic_labels


def compute_detection_metrics(gt_mask, pred_mask):
    """Precision, Recall, F1"""
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }


def get_gt_obstacle_mask(semantic_labels):
    """Máscara de obstáculos según SemanticKITTI semantic-kitti.yaml"""
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles (car, bicycle, bus, motorcycle, on-rails, truck, other-vehicle)
        30, 31, 32,                    # Persons (person, bicyclist, motorcyclist)
        50, 51, 52,                    # Structures (building, fence, other-structure)
        70, 71,                        # Vegetation (vegetation, trunk) — NO 72 (terrain = ground)
        80, 81,                        # Poles/signs (pole, traffic-sign)
        99,                            # other-object
        252, 253, 254, 255, 256, 257, 258, 259  # Moving (car, person, bicyclist, etc.)
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


def print_metrics(name, metrics, timing_ms=None):
    """Imprimir métricas formateadas"""
    print(f"  {name}:")
    print(f"    Precision: {100*metrics['precision']:.2f}%")
    print(f"    Recall:    {100*metrics['recall']:.2f}%")
    print(f"    F1 Score:  {100*metrics['f1']:.2f}%")
    print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    if timing_ms is not None:
        print(f"    Timing: {timing_ms:.0f} ms")


# ========================================
# TEST POR SECUENCIA
# ========================================

def test_sequence(seq: str, scan_start: int, n_frames: int):
    """Ejecutar test completo en una secuencia"""
    info = get_sequence_info(seq)

    print("=" * 80)
    print(f"SECUENCIA {seq} | Frame {scan_start}")
    print("=" * 80)

    # Verificar que existen los datos
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    # Frame de referencia
    points_ref, labels_ref = load_kitti_scan(scan_start, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)
    print(f"  Frame referencia (scan {scan_start}): {len(points_ref)} pts, {gt_mask.sum()} obstacles GT")
    print()

    results = {}

    # ========================================
    # 1. STAGE 2 SOLO (Baseline single-frame)
    # ========================================
    print("-" * 60)
    print("CONFIG 1: Stage 2 solo (baseline)")
    print("-" * 60)

    pipeline_s2 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        verbose=False
    ))
    result_s2 = pipeline_s2.stage2_complete(points_ref)
    metrics_s2 = compute_detection_metrics(gt_mask, result_s2['obs_mask'])
    print_metrics("Stage 2", metrics_s2, result_s2['timing_total_ms'])
    results['stage2'] = metrics_s2
    print()

    # ========================================
    # 2. STAGE 3 = Stage 2 + DBSCAN Cluster Filtering
    # ========================================
    print("-" * 60)
    print(f"CONFIG 2: Stage 3 = Stage 2 + DBSCAN")
    print("-" * 60)

    pipeline_s3 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.5,
        cluster_min_samples=5,
        cluster_min_pts=15,
        verbose=False
    ))

    result_s3 = pipeline_s3.stage3_complete(points_ref)

    n_clusters = result_s3.get('n_clusters', 0)
    n_removed = result_s3.get('n_cluster_total_removed', 0)
    print(f"  {result_s3['obs_mask'].sum()} obs | {n_clusters} clusters | {n_removed} pts removed")

    metrics_s3 = compute_detection_metrics(gt_mask, result_s3['obs_mask'])
    print_metrics("Stage 3", metrics_s3, result_s3.get('timing_total_ms'))
    print(f"    Stage 3 timing: {result_s3.get('timing_stage3_ms', 0):.1f} ms")
    print(f"    Clusters: {result_s3.get('n_clusters', 0)} valid | {result_s3.get('n_clusters_rejected', 0)} rejected")
    print(f"    Removed: {result_s3.get('n_cluster_total_removed', 0)} pts (noise: {result_s3.get('n_noise_removed', 0)}, small: {result_s3.get('n_small_cluster_removed', 0)})")
    results['stage3'] = metrics_s3
    print()

    # ========================================
    # 3. ABLATION: Stage 3 con diferentes min_pts
    # ========================================
    print("-" * 60)
    print(f"ABLATION: Stage 3 variando cluster_min_pts")
    print("-" * 60)

    for min_pts in [5, 10, 15, 25, 50]:
        pipeline_abl = LidarPipelineSuite(PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_cluster_filtering=True,
            cluster_eps=0.5,
            cluster_min_samples=5,
            cluster_min_pts=min_pts,
            verbose=False
        ))

        result_abl = pipeline_abl.stage3_complete(points_ref)

        m = compute_detection_metrics(gt_mask, result_abl['obs_mask'])
        fp_vs_s2 = results['stage2']['fp'] - m['fp']
        recall_loss = 100 * (results['stage2']['recall'] - m['recall'])
        print(f"  min_pts={min_pts:>3}: P={100*m['precision']:.1f}% R={100*m['recall']:.1f}% F1={100*m['f1']:.1f}% | FP removed: {fp_vs_s2} | Recall loss: {recall_loss:.2f}%")
        results[f'stage3_min{min_pts}'] = m

    print()

    # ========================================
    # COMPARATIVA
    # ========================================
    print("=" * 60)
    print(f"COMPARATIVA SECUENCIA {seq}")
    print("=" * 60)
    print()
    print(f"{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8}")
    print("-" * 71)
    for name in ['stage2', 'stage3']:
        if name in results:
            m = results[name]
            print(f"{name:<25} {100*m['precision']:>9.2f}% {100*m['recall']:>9.2f}% {100*m['f1']:>9.2f}% {m['fp']:>8} {m['fn']:>8}")
    print()

    # Impacto de DBSCAN
    if 'stage2' in results and 'stage3' in results:
        fp_reduction = results['stage2']['fp'] - results['stage3']['fp']
        fp_pct = 100 * fp_reduction / max(results['stage2']['fp'], 1)
        recall_loss = 100 * (results['stage2']['recall'] - results['stage3']['recall'])
        precision_gain = 100 * (results['stage3']['precision'] - results['stage2']['precision'])
        f1_change = 100 * (results['stage3']['f1'] - results['stage2']['f1'])

        print(f"  IMPACTO DBSCAN Cluster Filtering (Stage 2 -> Stage 3):")
        print(f"    FP eliminados: {fp_reduction} ({fp_pct:.1f}%)")
        print(f"    Precision:  {precision_gain:+.2f}%")
        print(f"    Recall:     {recall_loss:+.2f}% (negativo = pérdida)")
        print(f"    F1 Score:   {f1_change:+.2f}%")
        print()

    return results


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Test Stage 3 DBSCAN Cluster Filtering')
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
    if len(all_results) > 1:
        print()
        print("=" * 80)
        print("RESUMEN GLOBAL")
        print("=" * 80)
        for seq, res in all_results.items():
            if res is None:
                continue
            print(f"\n  Secuencia {seq}:")
            print(f"    Stage 2:    F1={100*res['stage2']['f1']:.1f}%  P={100*res['stage2']['precision']:.1f}%  R={100*res['stage2']['recall']:.1f}%")
            print(f"    Stage 3:    F1={100*res['stage3']['f1']:.1f}%  P={100*res['stage3']['precision']:.1f}%  R={100*res['stage3']['recall']:.1f}%")


if __name__ == '__main__':
    main()
