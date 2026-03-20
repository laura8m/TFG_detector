#!/usr/bin/env python3
"""
Grid Search PARALELO de parámetros delta-r + DBSCAN (Stages 2-3).

Diseñado para servidor multi-core (ej: AMD EPYC 7763 128C/256T).

Auto-descubre TODAS las secuencias SemanticKITTI disponibles (00-10) y usa
TODOS los frames que tengan tanto .bin como .label.

OPTIMIZACIÓN:
  1. Precomputa Stage 1 (Patchwork++) + delta_r continuo una sola vez (secuencial)
  2. Paraleliza la evaluación de combos con multiprocessing.Pool
  3. DBSCAN con voxel downsampling para reducir puntos

Uso:
    # Todas las secuencias, todos los frames, todos los cores
    python3 tests/test_delta_r_dbscan_grid_search.py --mode full

    # Submuestrear 1 de cada 5 frames para ir más rápido
    python3 tests/test_delta_r_dbscan_grid_search.py --mode full --stride 5

    # Solo delta-r (instantáneo, no necesita paralelizar)
    python3 tests/test_delta_r_dbscan_grid_search.py --mode delta_r

    # Controlar workers
    python3 tests/test_delta_r_dbscan_grid_search.py --mode full --workers 64

    # Solo algunas secuencias
    python3 tests/test_delta_r_dbscan_grid_search.py --seq 00 04 05
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product
from multiprocessing import Pool, cpu_count
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir,
                        get_labels_dir, VELODYNE_ROOT, LABELS_ROOT)


# ========================================
# AUTO-DESCUBRIMIENTO DE DATOS
# ========================================

def discover_sequences():
    """Encuentra secuencias que tienen tanto velodyne como labels."""
    if not VELODYNE_ROOT.exists() or not LABELS_ROOT.exists():
        return []
    vel_seqs = {d.name for d in VELODYNE_ROOT.iterdir() if d.is_dir()}
    lab_seqs = {d.name for d in LABELS_ROOT.iterdir() if d.is_dir()}
    seqs = sorted(vel_seqs & lab_seqs)
    return seqs


def discover_scan_ids(seq, stride=1):
    """Encuentra scan IDs que tienen tanto .bin como .label para una secuencia."""
    vel_dir = get_velodyne_dir(seq)
    lab_dir = get_labels_dir(seq)
    if not vel_dir.exists() or not lab_dir.exists():
        return []
    vel_ids = {int(f.stem) for f in vel_dir.glob('*.bin')}
    lab_ids = {int(f.stem) for f in lab_dir.glob('*.label')}
    all_ids = sorted(vel_ids & lab_ids)
    return all_ids[::stride]


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


OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    50, 51, 52,
    70, 71,
    80, 81,
    99,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)


def get_gt_obstacle_mask(semantic_labels):
    """SemanticKITTI obstacle labels (NO 72=terrain, SI 252-259=moving)"""
    return np.isin(semantic_labels, OBSTACLE_LABELS)


def compute_metrics_accum(gt_mask, pred_mask):
    """Retorna tp, fp, fn como enteros para acumulación."""
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    return tp, fp, fn


def metrics_from_accum(tp, fp, fn):
    """Calcula precision, recall, f1, iou desde acumulados."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn}


# ========================================
# GRIDS DE PARÁMETROS
# ========================================

def get_delta_r_grid():
    return {
        'threshold_obs':  [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        'threshold_void': [0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
    }


def get_dbscan_grid():
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
    Returns: list of dicts con: points, delta_r, rejected_wall_indices, gt_mask, n_points
    """
    config = PipelineConfig(enable_cluster_filtering=False, verbose=False)
    pipeline = LidarPipelineSuite(config)

    frames = []
    for scan_id in scan_ids:
        pts, labels = load_kitti_scan(scan_id, seq)
        gt_mask = get_gt_obstacle_mask(labels)

        stage1_result = pipeline.stage1_complete(pts)

        N = len(pts)
        n_per_point = stage1_result['n_per_point']
        d_per_point = stage1_result['d_per_point']
        r_measured = np.linalg.norm(pts, axis=1)
        dot_prod = np.einsum('ij,ij->i', pts, n_per_point) / np.maximum(r_measured, 1e-6)
        safe_dot = np.where(dot_prod < -1e-3, dot_prod, -1e-3)
        delta_r = np.clip(r_measured + d_per_point / safe_dot, -20.0, 10.0)

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
# REPLAY FUNCTIONS
# ========================================

def replay_delta_r_single(frame, threshold_obs, threshold_void):
    """Reclasifica un frame con nuevos umbrales. Retorna obs_mask."""
    obs_mask = (frame['delta_r'] < threshold_obs) | (frame['delta_r'] > threshold_void)
    if len(frame['rejected_wall_indices']) > 0:
        obs_mask[frame['rejected_wall_indices']] = True
    return obs_mask


def replay_dbscan(points, obs_mask, eps, min_samples, min_pts):
    """Ejecuta DBSCAN con voxel downsampling sobre puntos obstáculo."""
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

    # DBSCAN (n_jobs=1 porque ya paralelizamos por combos)
    db = DBSCAN(eps=eps, min_samples=max(2, min_samples // 2), n_jobs=1)
    voxel_labels = db.fit_predict(voxel_centroids)

    cluster_labels_obs = voxel_labels[inverse]

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
# DATOS GLOBALES PARA MULTIPROCESSING (fork)
# ========================================

_GLOBAL_FRAMES = None  # Se rellena antes de Pool


def _init_worker(frames):
    """Inicializa variable global en cada worker (fork comparte memoria)."""
    global _GLOBAL_FRAMES
    _GLOBAL_FRAMES = frames


# ========================================
# WORKERS PARA POOL
# ========================================

def _eval_delta_r_combo(args):
    """Worker: evalúa un combo delta-r sobre todos los frames."""
    threshold_obs, threshold_void = args
    total_tp, total_fp, total_fn = 0, 0, 0
    for frame in _GLOBAL_FRAMES:
        obs_mask = replay_delta_r_single(frame, threshold_obs, threshold_void)
        tp, fp, fn = compute_metrics_accum(frame['gt_mask'], obs_mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    m = metrics_from_accum(total_tp, total_fp, total_fn)
    return {
        'params': {'threshold_obs': threshold_obs, 'threshold_void': threshold_void},
        **m,
    }


def _eval_dbscan_combo(args):
    """Worker: evalúa un combo DBSCAN (con delta-r fijos) sobre todos los frames."""
    threshold_obs, threshold_void, eps, min_samples, min_pts = args
    total_tp, total_fp, total_fn = 0, 0, 0
    for frame in _GLOBAL_FRAMES:
        obs_mask = replay_delta_r_single(frame, threshold_obs, threshold_void)
        obs_filtered = replay_dbscan(frame['points'], obs_mask, eps, min_samples, min_pts)
        tp, fp, fn = compute_metrics_accum(frame['gt_mask'], obs_filtered)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    m = metrics_from_accum(total_tp, total_fp, total_fn)
    return {
        'params': {
            'threshold_obs': threshold_obs, 'threshold_void': threshold_void,
            'cluster_eps': eps, 'cluster_min_samples': min_samples, 'cluster_min_pts': min_pts,
        },
        **m,
    }


def _eval_full_combo(args):
    """Worker: evalúa un combo completo (delta-r + DBSCAN)."""
    return _eval_dbscan_combo(args)


# ========================================
# BÚSQUEDAS PARALELAS
# ========================================

def search_delta_r_parallel(frames, grid, n_workers):
    """Grid search paralelo de umbrales delta-r."""
    combos = list(product(grid['threshold_obs'], grid['threshold_void']))
    print(f"  [Delta-r] {len(combos)} combos × {len(frames)} frames, {n_workers} workers")

    t0 = time.time()
    # Delta-r es tan rápido que paralelizar puede tener más overhead que beneficio
    # pero con miles de frames sí merece la pena
    with Pool(n_workers, initializer=_init_worker, initargs=(frames,)) as pool:
        results = pool.map(_eval_delta_r_combo, combos, chunksize=max(1, len(combos) // n_workers))

    t_total = time.time() - t0
    print(f"  [Delta-r] {len(combos)} combos en {t_total:.1f}s")

    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def search_dbscan_parallel(frames, delta_r_params, dbscan_grid, n_workers):
    """Grid search paralelo de DBSCAN con delta-r fijos."""
    thr_obs = delta_r_params['threshold_obs']
    thr_void = delta_r_params['threshold_void']

    combos = [
        (thr_obs, thr_void, eps, ms, mp)
        for eps, ms, mp in product(
            dbscan_grid['cluster_eps'],
            dbscan_grid['cluster_min_samples'],
            dbscan_grid['cluster_min_pts'],
        )
    ]
    print(f"  [DBSCAN] {len(combos)} combos × {len(frames)} frames, {n_workers} workers")

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(frames,)) as pool:
        results = list(pool.imap_unordered(
            _eval_dbscan_combo, combos,
            chunksize=max(1, len(combos) // (n_workers * 4))
        ))

    t_total = time.time() - t0
    print(f"  [DBSCAN] {len(combos)} combos en {t_total:.1f}s ({t_total/len(combos):.2f}s/combo)")

    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def search_full_parallel(frames, delta_r_grid, dbscan_grid, n_workers):
    """Grid search paralelo combinado: delta-r × DBSCAN."""
    combos = [
        (thr_obs, thr_void, eps, ms, mp)
        for thr_obs, thr_void, eps, ms, mp in product(
            delta_r_grid['threshold_obs'],
            delta_r_grid['threshold_void'],
            dbscan_grid['cluster_eps'],
            dbscan_grid['cluster_min_samples'],
            dbscan_grid['cluster_min_pts'],
        )
    ]
    n_combos = len(combos)
    print(f"  [Full] {n_combos} combos × {len(frames)} frames, {n_workers} workers")

    t0 = time.time()
    done = [0]

    with Pool(n_workers, initializer=_init_worker, initargs=(frames,)) as pool:
        results = []
        chunksize = max(1, n_combos // (n_workers * 4))
        for r in pool.imap_unordered(_eval_full_combo, combos, chunksize=chunksize):
            results.append(r)
            done[0] += 1
            if done[0] % max(1, n_combos // 20) == 0 or done[0] == n_combos:
                elapsed = time.time() - t0
                eta = (elapsed / done[0]) * (n_combos - done[0])
                print(f"\r  [Full] {done[0]}/{n_combos} ({100*done[0]/n_combos:.0f}%) "
                      f"ETA: {eta:.0f}s", end="", flush=True)

    t_total = time.time() - t0
    print(f"\r  [Full] {n_combos} combos en {t_total:.1f}s "
          f"({t_total/n_combos:.3f}s/combo)" + " " * 30)

    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


# ========================================
# IMPRESIÓN DE RESULTADOS
# ========================================

def print_results(results, baseline_f1, baseline_iou, top_n, title, mode):
    """Imprime tabla de resultados."""
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"Baseline: F1={100*baseline_f1:.2f}%  IoU={100*baseline_iou:.2f}%")
    print(f"{'='*120}")

    if mode == 'delta_r':
        print(f"{'#':>4} {'thr_obs':>8} {'thr_void':>9} | {'F1':>8} {'IoU':>8} "
              f"{'dF1':>8} {'P':>8} {'R':>8} {'FP':>10} {'FN':>10}")
        print("-" * 100)
        for i, r in enumerate(results[:top_n]):
            p = r['params']
            df1 = r['f1'] - baseline_f1
            marker = " *" if df1 > 0.0001 else ""
            print(f"{i+1:>4} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                  f" | {100*r['f1']:>7.2f}% {100*r['iou']:>7.2f}% {100*df1:>+7.2f}%"
                  f" {100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}%"
                  f" {r['fp']:>10} {r['fn']:>10}{marker}")
    else:
        print(f"{'#':>4} {'thr_obs':>8} {'thr_void':>9} {'eps':>6} {'ms':>4} {'mp':>4}"
              f" | {'F1':>8} {'IoU':>8} {'dF1':>8} {'P':>8} {'R':>8} {'FP':>10} {'FN':>10}")
        print("-" * 120)
        for i, r in enumerate(results[:top_n]):
            p = r['params']
            df1 = r['f1'] - baseline_f1
            marker = " *" if df1 > 0.0001 else ""
            print(f"{i+1:>4} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                  f" {p['cluster_eps']:>6.2f} {p['cluster_min_samples']:>4} {p['cluster_min_pts']:>4}"
                  f" | {100*r['f1']:>7.2f}% {100*r['iou']:>7.2f}% {100*df1:>+7.2f}%"
                  f" {100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}%"
                  f" {r['fp']:>10} {r['fn']:>10}{marker}")

    n_better = sum(1 for r in results if r['f1'] > baseline_f1 + 0.0001)
    print(f"\n{n_better}/{len(results)} combinaciones mejoran vs baseline")

    if results and results[0]['f1'] > baseline_f1:
        best = results[0]
        p = best['params']
        params_str = ', '.join(f"{k}={v}" for k, v in p.items())
        df1 = best['f1'] - baseline_f1
        print(f"MEJOR: {params_str}")
        print(f"  F1={100*best['f1']:.2f}% ({100*df1:+.2f}%)  "
              f"IoU={100*best['iou']:.2f}%  P={100*best['precision']:.2f}%  R={100*best['recall']:.2f}%")


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search PARALELO delta-r + DBSCAN (todas las secuencias SemanticKITTI)')
    parser.add_argument('--seq', type=str, nargs='*', default=None,
                        help='Secuencias a usar (ej: 00 04 05). Default: todas las disponibles')
    parser.add_argument('--stride', type=int, default=1,
                        help='Usar 1 de cada N frames (default: 1 = todos)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['delta_r', 'dbscan', 'full'])
    parser.add_argument('--workers', type=int, default=0,
                        help='Número de workers (default: cpu_count)')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else cpu_count()

    # Descubrir secuencias
    if args.seq:
        seqs = args.seq
    else:
        seqs = discover_sequences()
    if not seqs:
        print("ERROR: No se encontraron secuencias con velodyne + labels")
        return

    delta_r_grid = get_delta_r_grid()
    dbscan_grid = get_dbscan_grid()

    n_dr = 1
    for v in delta_r_grid.values():
        n_dr *= len(v)
    n_db = 1
    for v in dbscan_grid.values():
        n_db *= len(v)

    print("=" * 120)
    print("GRID SEARCH PARALELO — PARÁMETROS DELTA-R + DBSCAN")
    print("=" * 120)
    print(f"\nSecuencias: {seqs}")
    print(f"Stride: {args.stride} (1 de cada {args.stride} frames)")
    print(f"Workers: {n_workers}")
    print(f"Modo: {args.mode}")
    if args.mode in ('delta_r', 'full'):
        print(f"Delta-r: {n_dr} combos — {delta_r_grid}")
    if args.mode in ('dbscan', 'full'):
        print(f"DBSCAN: {n_db} combos — {dbscan_grid}")
    if args.mode == 'full':
        print(f"Total combos: {n_dr * n_db}")

    # ========================================
    # PRECOMPUTAR Stage 1 + delta_r (TODAS las secuencias)
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 1: PRECOMPUTACIÓN Stage 1 + delta_r")
    print(f"{'='*120}")

    all_frames = []
    seq_frame_counts = {}
    t0_total = time.time()

    for seq in seqs:
        scan_ids = discover_scan_ids(seq, stride=args.stride)
        if not scan_ids:
            print(f"  Seq {seq}: SKIP (sin datos)")
            continue

        print(f"\n  Seq {seq}: {len(scan_ids)} frames (de {scan_ids[0]} a {scan_ids[-1]}, stride={args.stride})")
        t0 = time.time()

        frames = precompute_stage1_and_delta_r(seq, scan_ids)
        n_pts = sum(f['n_points'] for f in frames)
        n_gt = sum(int(f['gt_mask'].sum()) for f in frames)
        elapsed = time.time() - t0
        fps = len(frames) / elapsed if elapsed > 0 else 0

        print(f"    OK ({elapsed:.1f}s, {fps:.1f} fps) | {n_pts:,} pts, {n_gt:,} GT obs")

        all_frames.extend(frames)
        seq_frame_counts[seq] = len(frames)

    total_frames = len(all_frames)
    t_precomp = time.time() - t0_total
    total_pts = sum(f['n_points'] for f in all_frames)
    total_gt = sum(int(f['gt_mask'].sum()) for f in all_frames)
    mem_estimate_mb = sum(
        f['points'].nbytes + f['delta_r'].nbytes + f['gt_mask'].nbytes + f['rejected_wall_indices'].nbytes
        for f in all_frames
    ) / 1e6

    print(f"\n  TOTAL: {total_frames} frames, {total_pts:,} pts, {total_gt:,} GT obs")
    print(f"  Precomputación: {t_precomp:.1f}s | RAM estimada: {mem_estimate_mb:.0f} MB")

    if total_frames == 0:
        print("ERROR: No hay frames disponibles")
        return

    # ========================================
    # BASELINES
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 2: BASELINES")
    print(f"{'='*120}")

    current_thr_obs = -0.5
    current_thr_void = 0.8
    current_eps = 0.8
    current_min_samples = 8
    current_min_pts = 30

    # Baseline sin DBSCAN
    bl_tp, bl_fp, bl_fn = 0, 0, 0
    baseline_obs_masks = []
    for f in all_frames:
        obs_mask = replay_delta_r_single(f, current_thr_obs, current_thr_void)
        baseline_obs_masks.append(obs_mask)
        tp, fp, fn = compute_metrics_accum(f['gt_mask'], obs_mask)
        bl_tp += tp
        bl_fp += fp
        bl_fn += fn
    baseline = metrics_from_accum(bl_tp, bl_fp, bl_fn)
    print(f"\n  Baseline (thr_obs={current_thr_obs}, thr_void={current_thr_void}, sin DBSCAN):")
    print(f"    F1={100*baseline['f1']:.2f}%  IoU={100*baseline['iou']:.2f}%  "
          f"P={100*baseline['precision']:.2f}%  R={100*baseline['recall']:.2f}%")

    # Baseline con DBSCAN
    bld_tp, bld_fp, bld_fn = 0, 0, 0
    for i, f in enumerate(all_frames):
        obs_filtered = replay_dbscan(
            f['points'], baseline_obs_masks[i],
            current_eps, current_min_samples, current_min_pts
        )
        tp, fp, fn = compute_metrics_accum(f['gt_mask'], obs_filtered)
        bld_tp += tp
        bld_fp += fp
        bld_fn += fn
    baseline_dbscan = metrics_from_accum(bld_tp, bld_fp, bld_fn)
    print(f"  Baseline con DBSCAN (eps={current_eps}, ms={current_min_samples}, mp={current_min_pts}):")
    print(f"    F1={100*baseline_dbscan['f1']:.2f}%  IoU={100*baseline_dbscan['iou']:.2f}%  "
          f"P={100*baseline_dbscan['precision']:.2f}%  R={100*baseline_dbscan['recall']:.2f}%")

    del baseline_obs_masks  # Liberar memoria

    # ========================================
    # GRID SEARCH
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 3: GRID SEARCH")
    print(f"{'='*120}")

    if args.mode == 'delta_r':
        results = search_delta_r_parallel(all_frames, delta_r_grid, n_workers)
        print_results(results, baseline['f1'], baseline['iou'], args.top,
                      f"TOP {args.top} DELTA-R (micro-avg, {total_frames} frames, {len(seq_frame_counts)} seqs)",
                      'delta_r')

    elif args.mode == 'dbscan':
        results = search_dbscan_parallel(
            all_frames,
            {'threshold_obs': current_thr_obs, 'threshold_void': current_thr_void},
            dbscan_grid, n_workers
        )
        print_results(results, baseline['f1'], baseline['iou'], args.top,
                      f"TOP {args.top} DBSCAN (micro-avg, {total_frames} frames, {len(seq_frame_counts)} seqs)",
                      'dbscan')

    elif args.mode == 'full':
        results = search_full_parallel(all_frames, delta_r_grid, dbscan_grid, n_workers)
        print_results(results, baseline_dbscan['f1'], baseline_dbscan['iou'], args.top,
                      f"TOP {args.top} FULL (micro-avg, {total_frames} frames, {len(seq_frame_counts)} seqs)",
                      'full')

    # ========================================
    # RESUMEN FINAL
    # ========================================
    t_total = time.time() - t0_total
    print(f"\n{'='*120}")
    print("RESUMEN")
    print(f"{'='*120}")
    print(f"  Secuencias: {list(seq_frame_counts.keys())}")
    print(f"  Frames por secuencia: {seq_frame_counts}")
    print(f"  Total frames: {total_frames}")
    print(f"  Baseline sin DBSCAN: F1={100*baseline['f1']:.2f}%  IoU={100*baseline['iou']:.2f}%")
    print(f"  Baseline con DBSCAN: F1={100*baseline_dbscan['f1']:.2f}%  IoU={100*baseline_dbscan['iou']:.2f}%")
    if results:
        best = results[0]
        ref_f1 = baseline_dbscan['f1'] if args.mode == 'full' else baseline['f1']
        df1 = best['f1'] - ref_f1
        p = best['params']
        print(f"  Mejor encontrado: F1={100*best['f1']:.2f}% ({100*df1:+.2f}%)")
        print(f"    Params: {', '.join(f'{k}={v}' for k, v in p.items())}")
    print(f"  Tiempo total: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == '__main__':
    main()
