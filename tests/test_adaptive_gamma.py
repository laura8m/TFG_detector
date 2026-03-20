#!/usr/bin/env python3
"""
Test: Ablation de parametros DBSCAN en Stage 3

Compara pipeline completo (Stages 1+2+3) con diferentes configuraciones
de DBSCAN cluster filtering:
- Config A: eps=0.5, min_pts=15 (default)
- Config B: eps=0.3, min_pts=10 (más restrictivo)
- Config C: eps=0.8, min_pts=20 (más permisivo)
- Config D: sin cluster filtering (Stage 2 solo)

Objetivo: Encontrar la configuración óptima de DBSCAN para cada tipo de
escenario (urbano seq 00 vs autopista seq 04).
"""

import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file

# ========================================
# UTILIDADES
# ========================================

def load_kitti_scan(scan_id: int, seq: str = '04'):
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
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def get_gt_obstacle_mask(semantic_labels):
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,
        30, 31, 32,
        50, 51, 52,
        70, 71,
        80, 81,
        99,
        252, 253, 254, 255, 256, 257, 258, 259
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


# ========================================
# TEST POR SECUENCIA
# ========================================

def run_pipeline(seq, scan_id, config, pipeline_name):
    """Ejecutar pipeline completo y devolver metricas"""
    pts, labels = load_kitti_scan(scan_id, seq)
    gt_mask = get_gt_obstacle_mask(labels)

    pipeline = LidarPipelineSuite(config)
    result = pipeline.stage3_complete(pts)

    metrics = compute_detection_metrics(gt_mask, result['obs_mask'])

    n_clusters = result.get('n_clusters', 0)
    n_removed = result.get('n_cluster_total_removed', 0)
    print(f"  {pipeline_name}: P={100*metrics['precision']:.1f}% R={100*metrics['recall']:.1f}% F1={100*metrics['f1']:.1f}% | FP={metrics['fp']} FN={metrics['fn']} | clusters={n_clusters} removed={n_removed}")

    return metrics, result


def test_sequence(seq, scan_id):
    info = get_sequence_info(seq)
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    print("=" * 80)
    print(f"SECUENCIA {seq} | Frame {scan_id}")
    print("=" * 80)

    results = {}

    # --- Stage 2 baseline ---
    print("\n--- Stage 2 solo (baseline, sin cluster filtering) ---")
    pipeline_s2 = LidarPipelineSuite(PipelineConfig(
        enable_cluster_filtering=False,
        verbose=False
    ))
    pts, labels = load_kitti_scan(scan_id, seq)
    gt_mask = get_gt_obstacle_mask(labels)
    r_s2 = pipeline_s2.stage2_complete(pts)
    m_s2 = compute_detection_metrics(gt_mask, r_s2['obs_mask'])
    print(f"  Stage 2:       P={100*m_s2['precision']:.1f}% R={100*m_s2['recall']:.1f}% F1={100*m_s2['f1']:.1f}% | FP={m_s2['fp']} FN={m_s2['fn']}")
    results['stage2'] = m_s2

    # --- Config A: default DBSCAN ---
    print(f"\n--- Stage 3: DBSCAN default (eps=0.5, min_pts=15) ---")
    config_default = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.5,
        cluster_min_samples=5,
        cluster_min_pts=15,
        verbose=False
    )
    m_default, _ = run_pipeline(seq, scan_id, config_default, "DBSCAN(0.5,15)")
    results['dbscan_default'] = m_default

    # --- Config B: restrictivo ---
    print(f"\n--- Stage 3: DBSCAN restrictivo (eps=0.3, min_pts=10) ---")
    config_restrictive = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.3,
        cluster_min_samples=5,
        cluster_min_pts=10,
        verbose=False
    )
    m_restrictive, _ = run_pipeline(seq, scan_id, config_restrictive, "DBSCAN(0.3,10)")
    results['dbscan_restrictive'] = m_restrictive

    # --- Config C: permisivo ---
    print(f"\n--- Stage 3: DBSCAN permisivo (eps=0.8, min_pts=20) ---")
    config_permissive = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.8,
        cluster_min_samples=5,
        cluster_min_pts=20,
        verbose=False
    )
    m_permissive, _ = run_pipeline(seq, scan_id, config_permissive, "DBSCAN(0.8,20)")
    results['dbscan_permissive'] = m_permissive

    # --- Config D: agresivo ---
    print(f"\n--- Stage 3: DBSCAN agresivo (eps=0.5, min_pts=25) ---")
    config_aggressive = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.5,
        cluster_min_samples=5,
        cluster_min_pts=25,
        verbose=False
    )
    m_aggressive, _ = run_pipeline(seq, scan_id, config_aggressive, "DBSCAN(0.5,25)")
    results['dbscan_aggressive'] = m_aggressive

    # --- Comparativa ---
    print(f"\n{'='*60}")
    print(f"COMPARATIVA SECUENCIA {seq}")
    print(f"{'='*60}")
    print(f"\n{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8}")
    print("-" * 71)
    for name, m in results.items():
        print(f"{name:<25} {100*m['precision']:>9.1f}% {100*m['recall']:>9.1f}% {100*m['f1']:>9.1f}% {m['fp']:>8} {m['fn']:>8}")

    # Impacto vs baseline
    if 'stage2' in results and 'dbscan_default' in results:
        fp_change = results['stage2']['fp'] - results['dbscan_default']['fp']
        recall_change = 100 * (results['dbscan_default']['recall'] - results['stage2']['recall'])
        f1_change = 100 * (results['dbscan_default']['f1'] - results['stage2']['f1'])
        print(f"\n  IMPACTO DBSCAN default vs Stage 2:")
        print(f"    FP: {fp_change:+d} ({'menos' if fp_change > 0 else 'mas'})")
        print(f"    Recall: {recall_change:+.2f}%")
        print(f"    F1: {f1_change:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test DBSCAN parameter ablation')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    args = parser.parse_args()

    all_results = {}

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', args.scan_start)

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', args.scan_start)

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("RESUMEN GLOBAL")
        print("=" * 80)
        for seq, res in all_results.items():
            if res is None:
                continue
            print(f"\n  Secuencia {seq}:")
            for name, m in res.items():
                print(f"    {name:<25} F1={100*m['f1']:.1f}%  P={100*m['precision']:.1f}%  R={100*m['recall']:.1f}%")


if __name__ == '__main__':
    main()
