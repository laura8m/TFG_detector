#!/usr/bin/env python3
"""
Grid Search de parámetros delta-r + DBSCAN (Stages 2-3).

OPTIMIZACIÓN: Precomputa Stage 1 (Patchwork++ + wall rejection) y el delta_r continuo
una sola vez. Luego replaya:
  - Reclasificación con distintos threshold_obs / threshold_void (instantáneo, solo máscaras)
  - DBSCAN con distintos eps / min_samples / min_pts (solo sobre puntos obstáculo)

Parámetros explorados:
  Stage 2 (delta-r):
    - threshold_obs:  umbral negativo para clasificar obstáculo
    - threshold_void: umbral positivo para clasificar void (también obstáculo)
  Stage 3 (DBSCAN):
    - cluster_eps:         distancia máxima entre puntos del mismo cluster
    - cluster_min_samples: densidad mínima para core point
    - cluster_min_pts:     puntos mínimos por cluster válido

Uso:
    python3 tests/test_delta_r_dbscan_grid_search.py --seq both --n_frames 10
    python3 tests/test_delta_r_dbscan_grid_search.py --seq 04 --mode delta_r --top 20
    python3 tests/test_delta_r_dbscan_grid_search.py --seq both --mode dbscan --top 15
    python3 tests/test_delta_r_dbscan_grid_search.py --seq both --mode full --top 10
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file


# ========================================
# DATOS
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


def get_gt_obstacle_mask(semantic_labels):
    """SemanticKITTI obstacle labels (NO 72=terrain, SI 252-259=moving)"""
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
# GRIDS DE PARÁMETROS
# ========================================

def get_delta_r_grid():
    """Grid para umbrales delta-r (Stage 2)."""
    return {
        'threshold_obs':  [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        'threshold_void': [0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
    }


def get_dbscan_grid():
    """Grid para DBSCAN (Stage 3)."""
    return {
        'cluster_eps':         [0.5, 0.6, 0.8, 1.0, 1.2],
        'cluster_min_samples': [3, 5, 8, 12],
        'cluster_min_pts':     [10, 20, 30, 50],
    }


# ========================================
# PRECOMPUTACIÓN
# ========================================

def precompute_stage1_and_delta_r(seq, scan_ids):
    """
    Ejecuta Stage 1 (Patchwork++) y calcula delta_r continuo para cada frame.
    También guarda rejected_wall_indices para reclasificación posterior.

    Returns:
        list of dicts con: points, delta_r, rejected_wall_indices, gt_mask
    """
    config = PipelineConfig(
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    frames = []
    for scan_id in scan_ids:
        pts, labels = load_kitti_scan(scan_id, seq)
        gt_mask = get_gt_obstacle_mask(labels)

        # Stage 1
        stage1_result = pipeline.stage1_complete(pts)

        # Calcular delta_r continuo (sin clasificar)
        N = len(pts)
        n_per_point = stage1_result['n_per_point']
        d_per_point = stage1_result['d_per_point']
        r_measured = np.linalg.norm(pts, axis=1)
        dot_prod = np.einsum('ij,ij->i', pts, n_per_point) / np.maximum(r_measured, 1e-6)
        safe_dot = np.where(dot_prod < -1e-3, dot_prod, -1e-3)
        delta_r = np.clip(r_measured + d_per_point / safe_dot, -20.0, 10.0)

        # Índices de paredes rechazadas
        wall_indices = stage1_result.get('rejected_walls', np.array([], dtype=np.int64))
        if hasattr(wall_indices, '__len__') and len(wall_indices) > 0:
            wall_indices = wall_indices[wall_indices < N]
        else:
            wall_indices = np.array([], dtype=np.int64)

        frames.append({
            'points': pts,
            'delta_r': delta_r,
            'rejected_wall_indices': wall_indices,
            'gt_mask': gt_mask,
            'n_points': N,
        })

    return frames


# ========================================
# REPLAY: RECLASIFICACIÓN DELTA-R
# ========================================

def replay_delta_r(frames, threshold_obs, threshold_void):
    """
    Reclasifica delta_r con nuevos umbrales. Instantáneo (~0.1ms por frame).
    Retorna obs_mask por frame.
    """
    results = []
    for fd in frames:
        delta_r = fd['delta_r']
        N = fd['n_points']
        obs_mask = (delta_r < threshold_obs) | (delta_r > threshold_void)
        # Forzar paredes rechazadas como obstáculos
        if len(fd['rejected_wall_indices']) > 0:
            obs_mask[fd['rejected_wall_indices']] = True
        results.append(obs_mask)
    return results


# ========================================
# REPLAY: DBSCAN
# ========================================

def replay_dbscan(points, obs_mask, eps, min_samples, min_pts):
    """
    Ejecuta DBSCAN con voxel downsampling sobre puntos obstáculo.
    Retorna obs_mask filtrado.
    """
    obs_indices = np.where(obs_mask)[0]
    n_obs = len(obs_indices)
    if n_obs == 0:
        return obs_mask.copy()

    obs_pts = points[obs_indices]

    # Voxel downsampling
    voxel_size = eps * 0.3
    vox_coords = np.floor(obs_pts / voxel_size).astype(np.int32)
    vox_keys = (vox_coords[:, 0].astype(np.int64) * 1000003 +
                vox_coords[:, 1].astype(np.int64) * 1009 +
                vox_coords[:, 2].astype(np.int64))

    unique_keys, inverse, counts = np.unique(vox_keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)

    voxel_centroids = np.column_stack([
        np.bincount(inverse, weights=obs_pts[:, d], minlength=n_voxels)
        for d in range(3)
    ]) / counts[:, np.newaxis]
    voxel_centroids = voxel_centroids.astype(np.float32)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=max(2, min_samples // 2), n_jobs=-1)
    voxel_labels = db.fit_predict(voxel_centroids)

    # Propagar a puntos originales
    cluster_labels_obs = voxel_labels[inverse]

    # Filtrar clusters pequeños
    max_label = cluster_labels_obs.max()
    N = len(points)
    if max_label >= 0:
        cluster_sizes = np.bincount(cluster_labels_obs[cluster_labels_obs >= 0],
                                     minlength=max_label + 1)
        point_cluster_size = np.where(
            cluster_labels_obs >= 0,
            cluster_sizes[cluster_labels_obs.clip(0)],
            0
        )
        valid_mask_obs = point_cluster_size >= min_pts
    else:
        valid_mask_obs = np.zeros(n_obs, dtype=bool)

    obs_mask_new = np.zeros(N, dtype=bool)
    obs_mask_new[obs_indices[valid_mask_obs]] = True
    return obs_mask_new


# ========================================
# BÚSQUEDAS
# ========================================

def search_delta_r(frames, grid, baseline_metrics, top_n):
    """Grid search solo de umbrales delta-r (sin DBSCAN)."""
    keys = list(grid.keys())
    values = list(grid.values())
    n_combos = 1
    for v in values:
        n_combos *= len(v)

    results = []
    t0 = time.time()

    for idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(idx, 1)) * (n_combos - idx)
            print(f"\r  [Delta-r] {idx+1}/{n_combos} (ETA: {eta:.0f}s)...", end="", flush=True)

        obs_masks = replay_delta_r(frames, params['threshold_obs'], params['threshold_void'])

        # Métricas promedio sobre todos los frames
        frame_metrics = []
        for i, fd in enumerate(frames):
            m = compute_metrics(fd['gt_mask'], obs_masks[i])
            frame_metrics.append(m)

        avg_f1 = np.mean([m['f1'] for m in frame_metrics])
        avg_iou = np.mean([m['iou'] for m in frame_metrics])
        avg_p = np.mean([m['precision'] for m in frame_metrics])
        avg_r = np.mean([m['recall'] for m in frame_metrics])
        total_fp = sum(m['fp'] for m in frame_metrics)
        total_fn = sum(m['fn'] for m in frame_metrics)

        results.append({
            'params': params,
            'avg_f1': avg_f1,
            'avg_iou': avg_iou,
            'avg_p': avg_p,
            'avg_r': avg_r,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'delta_f1': avg_f1 - baseline_metrics['avg_f1'],
        })

    t_total = time.time() - t0
    print(f"\r  [Delta-r] {n_combos} combos en {t_total:.1f}s ({1000*t_total/n_combos:.1f}ms/combo)" + " " * 30)

    results.sort(key=lambda x: x['avg_f1'], reverse=True)
    return results


def search_dbscan(frames, obs_masks_per_frame, grid, baseline_metrics, top_n):
    """Grid search solo de DBSCAN (con umbrales delta-r fijos)."""
    keys = list(grid.keys())
    values = list(grid.values())
    n_combos = 1
    for v in values:
        n_combos *= len(v)

    results = []
    t0 = time.time()

    for idx, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))

        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(idx, 1)) * (n_combos - idx)
            print(f"\r  [DBSCAN] {idx+1}/{n_combos} (ETA: {eta:.0f}s)...", end="", flush=True)

        frame_metrics = []
        for i, fd in enumerate(frames):
            obs_filtered = replay_dbscan(
                fd['points'], obs_masks_per_frame[i],
                params['cluster_eps'], params['cluster_min_samples'], params['cluster_min_pts']
            )
            m = compute_metrics(fd['gt_mask'], obs_filtered)
            frame_metrics.append(m)

        avg_f1 = np.mean([m['f1'] for m in frame_metrics])
        avg_iou = np.mean([m['iou'] for m in frame_metrics])
        avg_p = np.mean([m['precision'] for m in frame_metrics])
        avg_r = np.mean([m['recall'] for m in frame_metrics])
        total_fp = sum(m['fp'] for m in frame_metrics)
        total_fn = sum(m['fn'] for m in frame_metrics)

        results.append({
            'params': params,
            'avg_f1': avg_f1,
            'avg_iou': avg_iou,
            'avg_p': avg_p,
            'avg_r': avg_r,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'delta_f1': avg_f1 - baseline_metrics['avg_f1'],
        })

    t_total = time.time() - t0
    print(f"\r  [DBSCAN] {n_combos} combos en {t_total:.1f}s ({1000*t_total/n_combos:.2f}s/combo)" + " " * 30)

    results.sort(key=lambda x: x['avg_f1'], reverse=True)
    return results


def search_full(frames, delta_r_grid, dbscan_grid, baseline_metrics, top_n):
    """Grid search combinado: delta-r × DBSCAN."""
    all_keys = list(delta_r_grid.keys()) + list(dbscan_grid.keys())
    all_values = list(delta_r_grid.values()) + list(dbscan_grid.values())
    n_combos = 1
    for v in all_values:
        n_combos *= len(v)

    print(f"  Total combinaciones: {n_combos}")

    results = []
    t0 = time.time()

    for idx, combo in enumerate(product(*all_values)):
        params = dict(zip(all_keys, combo))

        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(idx, 1)) * (n_combos - idx)
            print(f"\r  [Full] {idx+1}/{n_combos} (ETA: {eta:.0f}s)...", end="", flush=True)

        # 1. Reclasificar delta-r
        obs_masks = replay_delta_r(frames, params['threshold_obs'], params['threshold_void'])

        # 2. DBSCAN
        frame_metrics = []
        for i, fd in enumerate(frames):
            obs_filtered = replay_dbscan(
                fd['points'], obs_masks[i],
                params['cluster_eps'], params['cluster_min_samples'], params['cluster_min_pts']
            )
            m = compute_metrics(fd['gt_mask'], obs_filtered)
            frame_metrics.append(m)

        avg_f1 = np.mean([m['f1'] for m in frame_metrics])
        avg_iou = np.mean([m['iou'] for m in frame_metrics])
        avg_p = np.mean([m['precision'] for m in frame_metrics])
        avg_r = np.mean([m['recall'] for m in frame_metrics])
        total_fp = sum(m['fp'] for m in frame_metrics)
        total_fn = sum(m['fn'] for m in frame_metrics)

        results.append({
            'params': params,
            'avg_f1': avg_f1,
            'avg_iou': avg_iou,
            'avg_p': avg_p,
            'avg_r': avg_r,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'delta_f1': avg_f1 - baseline_metrics['avg_f1'],
        })

    t_total = time.time() - t0
    print(f"\r  [Full] {n_combos} combos en {t_total:.1f}s ({1000*t_total/n_combos:.0f}ms/combo)" + " " * 30)

    results.sort(key=lambda x: x['avg_f1'], reverse=True)
    return results


# ========================================
# IMPRESIÓN DE RESULTADOS
# ========================================

def print_results_delta_r(results, baseline, top_n, title):
    """Imprime tabla de resultados para búsqueda delta-r."""
    print(f"\n  {'='*90}")
    print(f"  {title}")
    print(f"  Baseline: F1={100*baseline['avg_f1']:.1f}%  IoU={100*baseline['avg_iou']:.1f}%")
    print(f"  {'='*90}")
    print(f"  {'#':>3} {'thr_obs':>8} {'thr_void':>9} | {'F1':>7} {'IoU':>7} {'dF1':>7} {'P':>7} {'R':>7} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*85}")

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        marker = " *" if r['delta_f1'] > 0.001 else ""
        print(f"  {i+1:>3} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
              f" | {100*r['avg_f1']:>6.1f}% {100*r['avg_iou']:>6.1f}% {100*r['delta_f1']:>+6.2f}%"
              f" {100*r['avg_p']:>6.1f}% {100*r['avg_r']:>6.1f}% {r['total_fp']:>8} {r['total_fn']:>8}{marker}")


def print_results_dbscan(results, baseline, top_n, title):
    """Imprime tabla de resultados para búsqueda DBSCAN."""
    print(f"\n  {'='*105}")
    print(f"  {title}")
    print(f"  Baseline (sin DBSCAN): F1={100*baseline['avg_f1']:.1f}%  IoU={100*baseline['avg_iou']:.1f}%")
    print(f"  {'='*105}")
    print(f"  {'#':>3} {'eps':>6} {'min_s':>6} {'min_p':>6} | {'F1':>7} {'IoU':>7} {'dF1':>7} {'P':>7} {'R':>7} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*95}")

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        marker = " *" if r['delta_f1'] > 0.001 else ""
        print(f"  {i+1:>3} {p['cluster_eps']:>6.2f} {p['cluster_min_samples']:>6} {p['cluster_min_pts']:>6}"
              f" | {100*r['avg_f1']:>6.1f}% {100*r['avg_iou']:>6.1f}% {100*r['delta_f1']:>+6.2f}%"
              f" {100*r['avg_p']:>6.1f}% {100*r['avg_r']:>6.1f}% {r['total_fp']:>8} {r['total_fn']:>8}{marker}")


def print_results_full(results, baseline, top_n, title):
    """Imprime tabla de resultados para búsqueda combinada."""
    print(f"\n  {'='*120}")
    print(f"  {title}")
    print(f"  Baseline (actual): F1={100*baseline['avg_f1']:.1f}%  IoU={100*baseline['avg_iou']:.1f}%")
    print(f"  {'='*120}")
    print(f"  {'#':>3} {'thr_obs':>8} {'thr_void':>9} {'eps':>6} {'min_s':>6} {'min_p':>6}"
          f" | {'F1':>7} {'IoU':>7} {'dF1':>7} {'P':>7} {'R':>7} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*110}")

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        marker = " *" if r['delta_f1'] > 0.001 else ""
        print(f"  {i+1:>3} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
              f" {p['cluster_eps']:>6.2f} {p['cluster_min_samples']:>6} {p['cluster_min_pts']:>6}"
              f" | {100*r['avg_f1']:>6.1f}% {100*r['avg_iou']:>6.1f}% {100*r['delta_f1']:>+6.2f}%"
              f" {100*r['avg_p']:>6.1f}% {100*r['avg_r']:>6.1f}% {r['total_fp']:>8} {r['total_fn']:>8}{marker}")


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Grid Search delta-r + DBSCAN')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10, help='Número de frames a evaluar')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--mode', type=str, default='full',
                        choices=['delta_r', 'dbscan', 'full'],
                        help='delta_r: solo umbrales | dbscan: solo clustering | full: ambos')
    parser.add_argument('--top', type=int, default=15, help='Mostrar los N mejores resultados')
    args = parser.parse_args()

    delta_r_grid = get_delta_r_grid()
    dbscan_grid = get_dbscan_grid()

    n_delta_r = 1
    for v in delta_r_grid.values():
        n_delta_r *= len(v)
    n_dbscan = 1
    for v in dbscan_grid.values():
        n_dbscan *= len(v)

    print("=" * 100)
    print("GRID SEARCH - PARÁMETROS DELTA-R + DBSCAN")
    print("=" * 100)
    print(f"\nModo: {args.mode}")
    if args.mode in ('delta_r', 'full'):
        print(f"\nDelta-r ({n_delta_r} combos):")
        for key, values in delta_r_grid.items():
            print(f"  {key}: {values}")
    if args.mode in ('dbscan', 'full'):
        print(f"\nDBSCAN ({n_dbscan} combos):")
        for key, values in dbscan_grid.items():
            print(f"  {key}: {values}")
    if args.mode == 'full':
        print(f"\nTotal combinaciones: {n_delta_r * n_dbscan}")

    seqs = []
    if args.seq in ('04', 'both'):
        seqs.append('04')
    if args.seq in ('00', 'both'):
        seqs.append('00')

    all_seq_results = {}

    for seq in seqs:
        info = get_sequence_info(seq)
        if not info['data_dir'].exists():
            print(f"\n  [SKIP] Datos no encontrados: {info['data_dir']}")
            continue

        scan_ids = list(range(args.scan_start, args.scan_start + args.n_frames))

        print(f"\n{'='*100}")
        print(f"SECUENCIA {seq} | Frames {scan_ids[0]}-{scan_ids[-1]} ({args.n_frames} frames)")
        print(f"{'='*100}")

        # ========================================
        # PRECOMPUTAR Stage 1 + delta_r
        # ========================================
        print("\n  [Precomputo] Stage 1 (Patchwork++) + delta_r continuo...", end=" ", flush=True)
        t0_pre = time.time()
        frames = precompute_stage1_and_delta_r(seq, scan_ids)
        t_pre = time.time() - t0_pre
        n_pts_total = sum(fd['n_points'] for fd in frames)
        n_gt_total = sum(fd['gt_mask'].sum() for fd in frames)
        print(f"OK ({t_pre:.1f}s) | {n_pts_total} pts, {n_gt_total} GT obs")

        # ========================================
        # BASELINE: parámetros actuales (sin DBSCAN)
        # ========================================
        current_thr_obs = -0.5
        current_thr_void = 0.8
        baseline_obs_masks = replay_delta_r(frames, current_thr_obs, current_thr_void)
        baseline_frame_metrics = [compute_metrics(fd['gt_mask'], baseline_obs_masks[i])
                                   for i, fd in enumerate(frames)]
        baseline = {
            'avg_f1': np.mean([m['f1'] for m in baseline_frame_metrics]),
            'avg_iou': np.mean([m['iou'] for m in baseline_frame_metrics]),
            'avg_p': np.mean([m['precision'] for m in baseline_frame_metrics]),
            'avg_r': np.mean([m['recall'] for m in baseline_frame_metrics]),
        }
        print(f"  Baseline (thr_obs={current_thr_obs}, thr_void={current_thr_void}, sin DBSCAN):")
        print(f"    F1={100*baseline['avg_f1']:.1f}%  IoU={100*baseline['avg_iou']:.1f}%"
              f"  P={100*baseline['avg_p']:.1f}%  R={100*baseline['avg_r']:.1f}%")

        # ========================================
        # BASELINE con DBSCAN actual
        # ========================================
        current_eps = 0.8
        current_min_samples = 8
        current_min_pts = 30
        baseline_dbscan_metrics = []
        for i, fd in enumerate(frames):
            obs_filtered = replay_dbscan(fd['points'], baseline_obs_masks[i],
                                          current_eps, current_min_samples, current_min_pts)
            m = compute_metrics(fd['gt_mask'], obs_filtered)
            baseline_dbscan_metrics.append(m)
        baseline_with_dbscan = {
            'avg_f1': np.mean([m['f1'] for m in baseline_dbscan_metrics]),
            'avg_iou': np.mean([m['iou'] for m in baseline_dbscan_metrics]),
            'avg_p': np.mean([m['precision'] for m in baseline_dbscan_metrics]),
            'avg_r': np.mean([m['recall'] for m in baseline_dbscan_metrics]),
        }
        print(f"  Baseline con DBSCAN (eps={current_eps}, min_s={current_min_samples}, min_p={current_min_pts}):")
        print(f"    F1={100*baseline_with_dbscan['avg_f1']:.1f}%  IoU={100*baseline_with_dbscan['avg_iou']:.1f}%"
              f"  P={100*baseline_with_dbscan['avg_p']:.1f}%  R={100*baseline_with_dbscan['avg_r']:.1f}%")

        # ========================================
        # BÚSQUEDAS
        # ========================================
        seq_results = {'baseline': baseline, 'baseline_dbscan': baseline_with_dbscan}

        if args.mode == 'delta_r':
            results = search_delta_r(frames, delta_r_grid, baseline, args.top)
            print_results_delta_r(results, baseline, args.top,
                                   f"TOP {args.top} DELTA-R — SECUENCIA {seq}")
            seq_results['delta_r'] = results

            # Resumen
            n_better = sum(1 for r in results if r['delta_f1'] > 0.001)
            print(f"\n  {n_better}/{len(results)} combinaciones mejoran vs baseline")
            if results and results[0]['delta_f1'] > 0:
                best = results[0]
                print(f"  MEJOR: threshold_obs={best['params']['threshold_obs']}, "
                      f"threshold_void={best['params']['threshold_void']} "
                      f"→ F1={100*best['avg_f1']:.1f}% ({100*best['delta_f1']:+.2f}%)")

        elif args.mode == 'dbscan':
            results = search_dbscan(frames, baseline_obs_masks, dbscan_grid, baseline, args.top)
            print_results_dbscan(results, baseline, args.top,
                                  f"TOP {args.top} DBSCAN — SECUENCIA {seq}")
            seq_results['dbscan'] = results

            n_better = sum(1 for r in results if r['delta_f1'] > 0.001)
            print(f"\n  {n_better}/{len(results)} combinaciones mejoran vs baseline (sin DBSCAN)")

        elif args.mode == 'full':
            results = search_full(frames, delta_r_grid, dbscan_grid, baseline_with_dbscan, args.top)
            print_results_full(results, baseline_with_dbscan, args.top,
                                f"TOP {args.top} FULL — SECUENCIA {seq}")
            seq_results['full'] = results

            n_better = sum(1 for r in results if r['delta_f1'] > 0.001)
            print(f"\n  {n_better}/{len(results)} combinaciones mejoran vs baseline actual")
            if results and results[0]['delta_f1'] > 0:
                best = results[0]
                p = best['params']
                print(f"  MEJOR: thr_obs={p['threshold_obs']}, thr_void={p['threshold_void']}, "
                      f"eps={p['cluster_eps']}, min_s={p['cluster_min_samples']}, min_p={p['cluster_min_pts']} "
                      f"→ F1={100*best['avg_f1']:.1f}% ({100*best['delta_f1']:+.2f}%)")

        all_seq_results[seq] = seq_results

    # ========================================
    # RESUMEN GLOBAL (si ambas secuencias)
    # ========================================
    if len(all_seq_results) == 2 and args.mode in ('delta_r', 'full'):
        print(f"\n{'='*100}")
        print(f"RESUMEN GLOBAL — MEJOR POR MEDIA F1 (AMBAS SECUENCIAS)")
        print(f"{'='*100}")

        key = 'delta_r' if args.mode == 'delta_r' else 'full'

        # Indexar por parámetros
        results_04 = {str(r['params']): r for r in all_seq_results['04'][key]}
        results_00 = {str(r['params']): r for r in all_seq_results['00'][key]}

        bl_04 = all_seq_results['04']['baseline' if args.mode == 'delta_r' else 'baseline_dbscan']
        bl_00 = all_seq_results['00']['baseline' if args.mode == 'delta_r' else 'baseline_dbscan']
        bl_avg = (bl_04['avg_f1'] + bl_00['avg_f1']) / 2

        combined = []
        for pkey in results_04:
            if pkey in results_00:
                r04 = results_04[pkey]
                r00 = results_00[pkey]
                avg_f1 = (r04['avg_f1'] + r00['avg_f1']) / 2
                avg_iou = (r04['avg_iou'] + r00['avg_iou']) / 2
                combined.append({
                    'params': r04['params'],
                    'f1_04': r04['avg_f1'],
                    'f1_00': r00['avg_f1'],
                    'iou_04': r04['avg_iou'],
                    'iou_00': r00['avg_iou'],
                    'avg_f1': avg_f1,
                    'avg_iou': avg_iou,
                    'delta_avg': avg_f1 - bl_avg,
                })

        combined.sort(key=lambda x: x['avg_f1'], reverse=True)

        print(f"\n  Baseline: Seq04 F1={100*bl_04['avg_f1']:.1f}%  Seq00 F1={100*bl_00['avg_f1']:.1f}%  Media={100*bl_avg:.1f}%")

        if args.mode == 'delta_r':
            print(f"\n  {'#':>3} {'thr_obs':>8} {'thr_void':>9} | {'04 F1':>7} {'00 F1':>7} {'Media':>7} {'dMedia':>7} | {'04 IoU':>7} {'00 IoU':>7}")
            print(f"  {'-'*85}")
            for i, c in enumerate(combined[:args.top]):
                p = c['params']
                marker = " *" if c['delta_avg'] > 0.001 else ""
                print(f"  {i+1:>3} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                      f" | {100*c['f1_04']:>6.1f}% {100*c['f1_00']:>6.1f}% {100*c['avg_f1']:>6.1f}% {100*c['delta_avg']:>+6.2f}%"
                      f" | {100*c['iou_04']:>6.1f}% {100*c['iou_00']:>6.1f}%{marker}")
        else:
            print(f"\n  {'#':>3} {'thr_obs':>8} {'thr_void':>9} {'eps':>6} {'min_s':>6} {'min_p':>6}"
                  f" | {'04 F1':>7} {'00 F1':>7} {'Media':>7} {'dMedia':>7}")
            print(f"  {'-'*100}")
            for i, c in enumerate(combined[:args.top]):
                p = c['params']
                marker = " *" if c['delta_avg'] > 0.001 else ""
                print(f"  {i+1:>3} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                      f" {p['cluster_eps']:>6.2f} {p['cluster_min_samples']:>6} {p['cluster_min_pts']:>6}"
                      f" | {100*c['f1_04']:>6.1f}% {100*c['f1_00']:>6.1f}% {100*c['avg_f1']:>6.1f}% {100*c['delta_avg']:>+6.2f}%{marker}")

        n_better = sum(1 for c in combined if c['delta_avg'] > 0.001)
        print(f"\n  {n_better}/{len(combined)} combinaciones mejoran la media F1")

        if combined and combined[0]['delta_avg'] > 0:
            best = combined[0]
            print(f"\n  ÓPTIMO GLOBAL:")
            p = best['params']
            params_str = ', '.join(f"{k}={v}" for k, v in p.items())
            print(f"    {params_str}")
            print(f"    Seq04: F1={100*best['f1_04']:.1f}%  Seq00: F1={100*best['f1_00']:.1f}%  Media: {100*best['avg_f1']:.1f}% ({100*best['delta_avg']:+.2f}%)")
        else:
            print(f"\n  Los parámetros actuales ya son óptimos dentro del grid explorado.")


if __name__ == '__main__':
    main()
