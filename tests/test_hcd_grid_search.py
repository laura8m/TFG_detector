#!/usr/bin/env python3
"""
Grid search de parámetros HCD para encontrar la configuración óptima.

Prueba dos estrategias:
A) Per-point HCD (actual): tanh(z_rel / scale) > threshold → reclasificar
B) Per-bin HCD (ERASOR++): max_z - min_z en bin CZM > threshold → reclasificar

Para cada estrategia, barre:
- hcd_z_rel_scale: [0.1, 0.2, 0.3, 0.5] (solo estrategia A)
- hcd_reclassify_threshold: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


# ========================================
# ESTRATEGIA B: Per-bin HCD (ERASOR++ style)
# ========================================

def apply_perbin_hcd(pipeline, points, stage2_result, ground_indices, bin_height_threshold):
    """
    ERASOR++ style: para cada bin CZM, calcula DD = max_z - min_z.
    Si DD > threshold, reclasifica ground points de ese bin como obstáculo.
    """
    obs_mask = stage2_result['obs_mask'].copy()
    likelihood = stage2_result['likelihood'].copy()

    ground_pts = points[ground_indices]
    if len(ground_pts) == 0:
        return obs_mask

    # Obtener bin CZM de cada punto ground
    z_idx, r_idx, s_idx = pipeline.get_czm_bin(ground_pts[:, 0], ground_pts[:, 1])

    # Crear clave de bin
    valid = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)
    bin_keys = np.full(len(ground_pts), -1, dtype=np.int64)
    bin_keys[valid] = z_idx[valid] * 10000 + r_idx[valid] * 100 + s_idx[valid]

    # Para cada bin único, calcular DD = max_z - min_z
    unique_keys = np.unique(bin_keys[valid])
    reclassify_mask_ground = np.zeros(len(ground_pts), dtype=bool)

    # Vectorizado: sort by bin key, compute per-group stats
    valid_idx = np.where(valid)[0]
    valid_keys = bin_keys[valid_idx]
    valid_z = ground_pts[valid_idx, 2]

    sort_order = np.argsort(valid_keys)
    sorted_keys = valid_keys[sort_order]
    sorted_z = valid_z[sort_order]
    sorted_valid_idx = valid_idx[sort_order]

    # Encontrar límites de grupos
    unique_sorted, group_starts = np.unique(sorted_keys, return_index=True)
    group_ends = np.append(group_starts[1:], len(sorted_keys))

    for i in range(len(unique_sorted)):
        start = group_starts[i]
        end = group_ends[i]
        if end - start < 3:  # Bins con pocos puntos: ignorar
            continue
        z_slice = sorted_z[start:end]
        dd = z_slice.max() - z_slice.min()
        if dd > bin_height_threshold:
            # Reclasificar todos los ground points de este bin
            idx_in_ground = sorted_valid_idx[start:end]
            reclassify_mask_ground[idx_in_ground] = True

    # Aplicar: ground points reclasificados → obstáculo
    # Solo reclasificar los que actualmente son ground (no ya obstáculo)
    ground_global_mask = np.zeros(len(points), dtype=bool)
    ground_global_mask[ground_indices] = True

    reclass_global = np.zeros(len(points), dtype=bool)
    reclass_global[ground_indices[reclassify_mask_ground]] = True

    # Solo reclasificar los que Stage 2 clasificó como ground
    reclass_final = reclass_global & (~stage2_result['obs_mask'])
    obs_mask[reclass_final] = True

    return obs_mask


# ========================================
# GRID SEARCH
# ========================================

def run_hcd_grid_search(seqs, n_frames, scan_start):
    # Pre-cargar datos y calcular Stage 1 + Stage 2 sin HCD (baseline)
    print("Cargando datos y calculando Stage 2 baseline (sin HCD)...")
    data = {}  # {(seq, scan_id): (points, gt_mask, stage1_result, stage2_result_no_hcd)}

    baseline_pipeline = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,  # Sin HCD para baseline
        verbose=False
    ))

    for seq in seqs:
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, labels = load_kitti_scan(scan_id, seq)
            gt_mask = get_gt_obstacle_mask(labels)
            result_s2 = baseline_pipeline.stage2_complete(pts)
            # ground_indices viene en el resultado de stage2_complete (heredado de stage1)
            ground_indices = result_s2['ground_indices'].copy()
            data[(seq, scan_id)] = (pts, gt_mask, result_s2, ground_indices)
            print(f"  Seq {seq} frame {scan_id}: {pts.shape[0]} pts, {gt_mask.sum()} GT obs")

    # Baseline sin HCD
    print("\n--- Baseline Stage 2 (sin HCD) ---")
    baseline_metrics = {}
    for seq in seqs:
        all_tp, all_fp, all_fn = 0, 0, 0
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, gt_mask, r_s2, gi = data[(seq, scan_id)]
            m = compute_metrics(gt_mask, r_s2['obs_mask'])
            all_tp += m['tp']
            all_fp += m['fp']
            all_fn += m['fn']
        p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        baseline_metrics[seq] = {'precision': p, 'recall': r, 'f1': f1}
        print(f"  Seq {seq}: P={100*p:.1f}% R={100*r:.1f}% F1={100*f1:.1f}%")

    baseline_mean_f1 = np.mean([baseline_metrics[s]['f1'] for s in seqs])
    print(f"  Mean F1: {100*baseline_mean_f1:.1f}%")

    # =============================================
    # ESTRATEGIA A: Per-point HCD con varios thresholds y scales
    # =============================================
    print("\n" + "=" * 90)
    print("ESTRATEGIA A: Per-point HCD (tanh(z_rel / scale) > threshold)")
    print("=" * 90)

    scale_values = [0.1, 0.2, 0.3, 0.5]
    threshold_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results_a = []
    combos_a = list(product(scale_values, threshold_values))
    print(f"Combinaciones: {len(combos_a)}")

    for idx, (scale, thresh) in enumerate(combos_a):
        config = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=True,
            hcd_z_rel_scale=scale,
            hcd_reclassify_threshold=thresh,
            verbose=False
        )
        pipeline = LidarPipelineSuite(config)

        combo_results = {}
        n_reclassified_total = 0
        for seq in seqs:
            all_tp, all_fp, all_fn = 0, 0, 0
            for i in range(n_frames):
                scan_id = scan_start + i
                pts, gt_mask, _, _ = data[(seq, scan_id)]
                r = pipeline.stage2_complete(pts)
                m = compute_metrics(gt_mask, r['obs_mask'])
                all_tp += m['tp']
                all_fp += m['fp']
                all_fn += m['fn']

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            r_val = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
            combo_results[seq] = {'precision': p, 'recall': r_val, 'f1': f1}

        mean_f1 = np.mean([combo_results[s]['f1'] for s in seqs])
        mean_p = np.mean([combo_results[s]['precision'] for s in seqs])
        mean_r = np.mean([combo_results[s]['recall'] for s in seqs])
        delta = mean_f1 - baseline_mean_f1

        results_a.append({
            'scale': scale, 'threshold': thresh,
            'per_seq': combo_results,
            'mean_f1': mean_f1, 'mean_p': mean_p, 'mean_r': mean_r,
            'delta': delta
        })

        if (idx + 1) % 7 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(combos_a)}] scale={scale} thresh={thresh} -> F1={100*mean_f1:.2f}% (Δ={100*delta:+.2f}%)")

    results_a.sort(key=lambda x: x['mean_f1'], reverse=True)

    print(f"\nTOP 10 Per-point HCD:")
    print(f"{'Rank':<5} {'scale':>6} {'thresh':>7} | {'04 P%':>6} {'04 R%':>6} {'04 F1%':>7} | {'00 P%':>6} {'00 R%':>6} {'00 F1%':>7} | {'Mean F1':>8} {'Δ':>7}")
    print("-" * 90)
    for i, r in enumerate(results_a[:10]):
        s04 = r['per_seq'].get('04', {'precision': 0, 'recall': 0, 'f1': 0})
        s00 = r['per_seq'].get('00', {'precision': 0, 'recall': 0, 'f1': 0})
        print(f"{i+1:<5} {r['scale']:>6.1f} {r['threshold']:>7.1f} | {100*s04['precision']:>6.1f} {100*s04['recall']:>6.1f} {100*s04['f1']:>7.1f} | {100*s00['precision']:>6.1f} {100*s00['recall']:>6.1f} {100*s00['f1']:>7.1f} | {100*r['mean_f1']:>7.2f}% {100*r['delta']:>+6.2f}%")

    # =============================================
    # ESTRATEGIA B: Per-bin HCD (ERASOR++ style)
    # =============================================
    print("\n" + "=" * 90)
    print("ESTRATEGIA B: Per-bin HCD (DD = max_z - min_z por bin CZM)")
    print("=" * 90)

    bin_thresholds = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"Thresholds: {bin_thresholds}")

    # Necesitamos un pipeline con acceso a get_czm_bin
    ref_pipeline = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,
        verbose=False
    ))

    results_b = []
    for idx, bin_thresh in enumerate(bin_thresholds):
        combo_results = {}
        for seq in seqs:
            all_tp, all_fp, all_fn = 0, 0, 0
            for i in range(n_frames):
                scan_id = scan_start + i
                pts, gt_mask, r_s2, ground_indices = data[(seq, scan_id)]

                # Recalcular Stage 1 para tener acceso a get_czm_bin
                ref_pipeline.stage1_complete(pts)

                # Aplicar per-bin HCD
                obs_mask_hcd = apply_perbin_hcd(
                    ref_pipeline, pts, r_s2, ground_indices, bin_thresh
                )
                m = compute_metrics(gt_mask, obs_mask_hcd)
                all_tp += m['tp']
                all_fp += m['fp']
                all_fn += m['fn']

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            r_val = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
            combo_results[seq] = {'precision': p, 'recall': r_val, 'f1': f1}

        mean_f1 = np.mean([combo_results[s]['f1'] for s in seqs])
        mean_p = np.mean([combo_results[s]['precision'] for s in seqs])
        mean_r = np.mean([combo_results[s]['recall'] for s in seqs])
        delta = mean_f1 - baseline_mean_f1

        results_b.append({
            'bin_threshold': bin_thresh,
            'per_seq': combo_results,
            'mean_f1': mean_f1, 'mean_p': mean_p, 'mean_r': mean_r,
            'delta': delta
        })
        print(f"  [{idx+1}/{len(bin_thresholds)}] bin_thresh={bin_thresh:.2f} -> F1={100*mean_f1:.2f}% (Δ={100*delta:+.2f}%)")

    results_b.sort(key=lambda x: x['mean_f1'], reverse=True)

    print(f"\nResultados Per-bin HCD:")
    print(f"{'Rank':<5} {'thresh':>7} | {'04 P%':>6} {'04 R%':>6} {'04 F1%':>7} | {'00 P%':>6} {'00 R%':>6} {'00 F1%':>7} | {'Mean F1':>8} {'Δ':>7}")
    print("-" * 90)
    for i, r in enumerate(results_b):
        s04 = r['per_seq'].get('04', {'precision': 0, 'recall': 0, 'f1': 0})
        s00 = r['per_seq'].get('00', {'precision': 0, 'recall': 0, 'f1': 0})
        print(f"{i+1:<5} {r['bin_threshold']:>7.2f} | {100*s04['precision']:>6.1f} {100*s04['recall']:>6.1f} {100*s04['f1']:>7.1f} | {100*s00['precision']:>6.1f} {100*s00['recall']:>6.1f} {100*s00['f1']:>7.1f} | {100*r['mean_f1']:>7.2f}% {100*r['delta']:>+6.2f}%")

    # =============================================
    # RESUMEN FINAL
    # =============================================
    print("\n" + "=" * 90)
    print("RESUMEN COMPARATIVO")
    print("=" * 90)
    print(f"Baseline (sin HCD):          Mean F1 = {100*baseline_mean_f1:.2f}%")
    if results_a:
        best_a = results_a[0]
        print(f"Mejor Per-point HCD:         Mean F1 = {100*best_a['mean_f1']:.2f}% (scale={best_a['scale']}, thresh={best_a['threshold']}) Δ={100*best_a['delta']:+.2f}%")
    if results_b:
        best_b = results_b[0]
        print(f"Mejor Per-bin HCD (ERASOR++): Mean F1 = {100*best_b['mean_f1']:.2f}% (bin_thresh={best_b['bin_threshold']}) Δ={100*best_b['delta']:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description='HCD parameter grid search')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--scan_start', type=int, default=0)
    args = parser.parse_args()

    seqs = ['04', '00'] if args.seq == 'both' else [args.seq]
    run_hcd_grid_search(seqs, args.n_frames, args.scan_start)


if __name__ == '__main__':
    main()
