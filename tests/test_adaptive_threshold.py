#!/usr/bin/env python3
"""
Test: Umbral delta-r adaptativo por distancia vs fijo.

Compara:
- Baseline: threshold_obs = -0.5 (fijo)
- Adaptativo: threshold_obs(r) = -0.5 * (1 + k * r/r_max) con varios k

Evaluado en ambas secuencias (Seq 04 highway, Seq 00 urban).
"""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file


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


def run_test(seqs, n_frames, scan_start):
    # Configuraciones a probar
    configs = {}

    # Baseline: threshold fijo
    configs['Fijo (baseline)'] = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        adaptive_threshold=False,
        enable_cluster_filtering=False,
        verbose=False
    )

    # Adaptativo con distintos k
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        configs[f'Adaptativo k={k}'] = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            adaptive_threshold=True,
            adaptive_threshold_k=k,
            enable_cluster_filtering=False,
            verbose=False
        )

    # También probar con DBSCAN
    configs['Fijo + DBSCAN'] = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        adaptive_threshold=False,
        enable_cluster_filtering=True,
        verbose=False
    )

    for k in [1.0, 2.0, 3.0]:
        configs[f'Adapt k={k} + DBSCAN'] = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            adaptive_threshold=True,
            adaptive_threshold_k=k,
            enable_cluster_filtering=True,
            verbose=False
        )

    # Pre-cargar datos
    print("Cargando datos...")
    data = {}
    for seq in seqs:
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, labels = load_kitti_scan(scan_id, seq)
            gt_mask = get_gt_obstacle_mask(labels)
            data[(seq, scan_id)] = (pts, gt_mask)
            print(f"  Seq {seq} frame {scan_id}: {pts.shape[0]} pts, {gt_mask.sum()} GT obs")

    # Ejecutar cada configuración
    results = []
    for name, config in configs.items():
        pipeline = LidarPipelineSuite(config)
        combo_results = {}

        for seq in seqs:
            all_tp, all_fp, all_fn = 0, 0, 0
            for i in range(n_frames):
                scan_id = scan_start + i
                pts, gt_mask = data[(seq, scan_id)]

                if config.enable_cluster_filtering:
                    result = pipeline.stage3_complete(pts)
                else:
                    result = pipeline.stage2_complete(pts)

                m = compute_metrics(gt_mask, result['obs_mask'])
                all_tp += m['tp']
                all_fp += m['fp']
                all_fn += m['fn']

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            combo_results[seq] = {'precision': p, 'recall': r, 'f1': f1, 'fp': all_fp, 'fn': all_fn}

        mean_f1 = np.mean([combo_results[s]['f1'] for s in seqs])
        mean_p = np.mean([combo_results[s]['precision'] for s in seqs])
        mean_r = np.mean([combo_results[s]['recall'] for s in seqs])

        results.append({
            'name': name,
            'per_seq': combo_results,
            'mean_f1': mean_f1,
            'mean_p': mean_p,
            'mean_r': mean_r,
        })
        print(f"  {name}: Mean F1={100*mean_f1:.2f}% P={100*mean_p:.1f}% R={100*mean_r:.1f}%")

    # Resultados ordenados
    results.sort(key=lambda x: x['mean_f1'], reverse=True)

    print("\n" + "=" * 110)
    print("RESULTADOS (ordenados por Mean F1)")
    print("=" * 110)
    print(f"{'Rank':<5} {'Config':<25} | {'04 P%':>6} {'04 R%':>6} {'04 F1%':>7} | {'00 P%':>6} {'00 R%':>6} {'00 F1%':>7} | {'Mean F1':>8} {'Mean P':>8} {'Mean R':>8}")
    print("-" * 110)

    for i, r in enumerate(results):
        s04 = r['per_seq'].get('04', {'precision': 0, 'recall': 0, 'f1': 0})
        s00 = r['per_seq'].get('00', {'precision': 0, 'recall': 0, 'f1': 0})
        print(f"{i+1:<5} {r['name']:<25} | {100*s04['precision']:>6.1f} {100*s04['recall']:>6.1f} {100*s04['f1']:>7.1f} | {100*s00['precision']:>6.1f} {100*s00['recall']:>6.1f} {100*s00['f1']:>7.1f} | {100*r['mean_f1']:>7.2f}% {100*r['mean_p']:>7.1f}% {100*r['mean_r']:>7.1f}%")

    # Comparar mejor adaptativo vs baseline
    baseline = next(r for r in results if r['name'] == 'Fijo (baseline)')
    best_adapt = next((r for r in results if 'Adaptativo' in r['name']), None)
    baseline_dbscan = next((r for r in results if r['name'] == 'Fijo + DBSCAN'), None)
    best_adapt_dbscan = next((r for r in results if 'Adapt' in r['name'] and 'DBSCAN' in r['name']), None)

    print(f"\n--- Comparacion ---")
    print(f"Baseline fijo:           Mean F1 = {100*baseline['mean_f1']:.2f}%")
    if best_adapt:
        delta = best_adapt['mean_f1'] - baseline['mean_f1']
        print(f"Mejor adaptativo:        Mean F1 = {100*best_adapt['mean_f1']:.2f}% ({best_adapt['name']}) Δ={100*delta:+.2f}%")
    if baseline_dbscan:
        print(f"Fijo + DBSCAN:           Mean F1 = {100*baseline_dbscan['mean_f1']:.2f}%")
    if best_adapt_dbscan:
        delta = best_adapt_dbscan['mean_f1'] - baseline_dbscan['mean_f1']
        print(f"Mejor adapt + DBSCAN:    Mean F1 = {100*best_adapt_dbscan['mean_f1']:.2f}% ({best_adapt_dbscan['name']}) Δ={100*delta:+.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test adaptive delta-r threshold')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--scan_start', type=int, default=0)
    args = parser.parse_args()

    seqs = ['04', '00'] if args.seq == 'both' else [args.seq]
    run_test(seqs, args.n_frames, args.scan_start)
