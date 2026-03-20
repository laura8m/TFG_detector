#!/usr/bin/env python3
"""
Grid search de parametros DBSCAN (Stage 3) para encontrar la configuracion optima.

Barre combinaciones de:
- eps: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
- min_samples: [3, 5, 8]
- min_pts: [5, 10, 15, 20, 25, 30]

Evaluado en N frames de seq 00 y seq 04 con ground truth SemanticKITTI.
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file

# ========================================
# DATOS
# ========================================

def load_kitti_scan(scan_id, seq='04'):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    return points, semantic_labels


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


def compute_metrics(gt_mask, pred_mask):
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


# ========================================
# GRID SEARCH
# ========================================

def run_grid_search(seqs, n_frames, scan_start):
    # Rangos de parametros
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    min_samples_values = [3, 5, 8]
    min_pts_values = [5, 10, 15, 20, 25, 30]

    # Pre-cargar datos y calcular Stage 2 baseline (no depende de DBSCAN)
    print("Cargando datos y calculando Stage 2 baseline...")
    data = {}  # {(seq, scan_id): (points, gt_mask, stage2_result)}

    baseline_pipeline = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        verbose=False
    ))

    for seq in seqs:
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, labels = load_kitti_scan(scan_id, seq)
            gt_mask = get_gt_obstacle_mask(labels)
            result_s2 = baseline_pipeline.stage2_complete(pts)
            data[(seq, scan_id)] = (pts, gt_mask, result_s2)
            print(f"  Seq {seq} frame {scan_id}: {pts.shape[0]} pts, {gt_mask.sum()} GT obs")

    # Baseline Stage 2 (sin DBSCAN)
    print("\n--- Baseline Stage 2 (sin DBSCAN) ---")
    baseline_metrics = {}
    for seq in seqs:
        all_tp, all_fp, all_fn = 0, 0, 0
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, gt_mask, r_s2 = data[(seq, scan_id)]
            m = compute_metrics(gt_mask, r_s2['obs_mask'])
            all_tp += m['tp']
            all_fp += m['fp']
            all_fn += m['fn']
        p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        baseline_metrics[seq] = {'precision': p, 'recall': r, 'f1': f1}
        print(f"  Seq {seq}: P={100*p:.1f}% R={100*r:.1f}% F1={100*f1:.1f}%")

    # Grid search
    combos = list(product(eps_values, min_samples_values, min_pts_values))
    n_combos = len(combos)
    print(f"\nGrid search: {n_combos} combinaciones x {len(seqs)} seqs x {n_frames} frames = {n_combos * len(seqs) * n_frames} evaluaciones")

    results = []
    t_start = time.time()

    for idx, (eps, min_samples, min_pts) in enumerate(combos):
        config = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_cluster_filtering=True,
            cluster_eps=eps,
            cluster_min_samples=min_samples,
            cluster_min_pts=min_pts,
            verbose=False
        )
        pipeline = LidarPipelineSuite(config)

        combo_results = {}
        for seq in seqs:
            all_tp, all_fp, all_fn = 0, 0, 0
            for i in range(n_frames):
                scan_id = scan_start + i
                pts, gt_mask, r_s2 = data[(seq, scan_id)]
                # Aplicar Stage 3 sobre resultado Stage 2
                r_s3 = pipeline.stage3_cluster_filtering(pts, r_s2)
                m = compute_metrics(gt_mask, r_s3['obs_mask'])
                all_tp += m['tp']
                all_fp += m['fp']
                all_fn += m['fn']

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            combo_results[seq] = {'precision': p, 'recall': r, 'f1': f1, 'tp': all_tp, 'fp': all_fp, 'fn': all_fn}

        # Media F1 entre secuencias
        mean_f1 = np.mean([combo_results[s]['f1'] for s in seqs])
        mean_p = np.mean([combo_results[s]['precision'] for s in seqs])
        mean_r = np.mean([combo_results[s]['recall'] for s in seqs])

        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'min_pts': min_pts,
            'per_seq': combo_results,
            'mean_f1': mean_f1,
            'mean_precision': mean_p,
            'mean_recall': mean_r,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (n_combos - idx - 1)
            print(f"  [{idx+1}/{n_combos}] eps={eps} ms={min_samples} mp={min_pts} -> mean_F1={100*mean_f1:.1f}% (ETA: {eta:.0f}s)")

    # Ordenar por mean F1
    results.sort(key=lambda x: x['mean_f1'], reverse=True)

    # Resultados
    print("\n" + "=" * 110)
    print("TOP 20 CONFIGURACIONES (por mean F1)")
    print("=" * 110)
    header = f"{'Rank':<5} {'eps':>5} {'ms':>4} {'mp':>4}"
    for seq in seqs:
        header += f" | {seq} P%"
        header += f"  {seq} R%"
        header += f"  {seq} F1%"
    header += f" | {'Mean F1':>8} {'Mean P':>8} {'Mean R':>8}"
    print(header)
    print("-" * 110)

    for i, r in enumerate(results[:20]):
        line = f"{i+1:<5} {r['eps']:>5.1f} {r['min_samples']:>4} {r['min_pts']:>4}"
        for seq in seqs:
            s = r['per_seq'][seq]
            line += f" | {100*s['precision']:>5.1f} {100*s['recall']:>5.1f} {100*s['f1']:>5.1f}"
        line += f" | {100*r['mean_f1']:>7.1f}% {100*r['mean_precision']:>7.1f}% {100*r['mean_recall']:>7.1f}%"
        print(line)

    # Baseline comparacion
    baseline_mean_f1 = np.mean([baseline_metrics[s]['f1'] for s in seqs])
    print(f"\nBaseline Stage 2 (sin DBSCAN): mean F1 = {100*baseline_mean_f1:.1f}%")
    print(f"Mejor DBSCAN:                  mean F1 = {100*results[0]['mean_f1']:.1f}% (eps={results[0]['eps']}, ms={results[0]['min_samples']}, mp={results[0]['min_pts']})")
    delta = results[0]['mean_f1'] - baseline_mean_f1
    print(f"Mejora:                         +{100*delta:.2f}%")

    # Worst 5 (para ver que evitar)
    print(f"\nBOTTOM 5 (peores configuraciones):")
    for r in results[-5:]:
        print(f"  eps={r['eps']:.1f} ms={r['min_samples']} mp={r['min_pts']} -> mean_F1={100*r['mean_f1']:.1f}%")

    elapsed_total = time.time() - t_start
    print(f"\nTiempo total: {elapsed_total:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='DBSCAN parameter grid search')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--scan_start', type=int, default=0)
    args = parser.parse_args()

    seqs = ['04', '00'] if args.seq == 'both' else [args.seq]
    run_grid_search(seqs, args.n_frames, args.scan_start)


if __name__ == '__main__':
    main()
