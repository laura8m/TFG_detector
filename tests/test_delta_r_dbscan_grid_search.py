#!/usr/bin/env python3
"""
Grid Search PARALELO de parámetros delta-r + DBSCAN (Stages 2-3).

Protocolo SemanticKITTI reproducible:
  - TRAIN (tuning):  seq 00-07, 09-10  (~19k frames)
  - VAL (evaluación): seq 08            (~4k frames)
  Las secuencias 11-21 son test set sin labels públicas.

Auto-descubre secuencias disponibles. Separa train/val automáticamente.
Optimiza en train, evalúa los mejores en val, y reporta ambos.

Diseñado para servidor multi-core (ej: AMD EPYC 7763 128C/256T).

Uso:
    # Protocolo completo: tuning en train, evaluación en val (seq 08)
    python3 tests/test_delta_r_dbscan_grid_search.py --mode full --workers 128 --stride 5

    # Solo delta-r (rápido)
    python3 tests/test_delta_r_dbscan_grid_search.py --mode delta_r --stride 5

    # Cambiar secuencia de validación
    python3 tests/test_delta_r_dbscan_grid_search.py --val_seq 08 --stride 5

    # Forzar secuencias manualmente (sin split automático)
    python3 tests/test_delta_r_dbscan_grid_search.py --train_seq 00 04 --val_seq 08
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

# Split oficial SemanticKITTI
SEMANTICKITTI_TRAIN = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
SEMANTICKITTI_VAL = ['08']


# ========================================
# AUTO-DESCUBRIMIENTO DE DATOS
# ========================================

def discover_sequences():
    """Encuentra secuencias que tienen tanto velodyne como labels con archivos reales."""
    if not VELODYNE_ROOT.exists() or not LABELS_ROOT.exists():
        return []
    vel_seqs = {d.name for d in VELODYNE_ROOT.iterdir() if d.is_dir()}
    lab_seqs = set()
    for d in LABELS_ROOT.iterdir():
        if d.is_dir():
            lab_dir = d / "labels"
            if lab_dir.exists() and any(lab_dir.glob('*.label')):
                lab_seqs.add(d.name)
    return sorted(vel_seqs & lab_seqs)


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


def precompute_all_sequences(seqs, stride):
    """Precomputa Stage 1 + delta_r para múltiples secuencias."""
    all_frames = []
    seq_counts = {}
    t0 = time.time()

    for seq in seqs:
        scan_ids = discover_scan_ids(seq, stride=stride)
        if not scan_ids:
            print(f"  Seq {seq}: SKIP (sin datos)")
            continue

        print(f"  Seq {seq}: {len(scan_ids)} frames "
              f"(de {scan_ids[0]} a {scan_ids[-1]}, stride={stride})...", end=" ", flush=True)
        t_seq = time.time()

        frames = precompute_stage1_and_delta_r(seq, scan_ids)
        n_pts = sum(f['n_points'] for f in frames)
        n_gt = sum(int(f['gt_mask'].sum()) for f in frames)
        elapsed = time.time() - t_seq
        fps = len(frames) / elapsed if elapsed > 0 else 0

        print(f"OK ({elapsed:.1f}s, {fps:.1f} fps) | {n_pts:,} pts, {n_gt:,} GT obs")

        all_frames.extend(frames)
        seq_counts[seq] = len(frames)

    t_total = time.time() - t0
    total_frames = len(all_frames)
    mem_mb = sum(
        f['points'].nbytes + f['delta_r'].nbytes + f['gt_mask'].nbytes + f['rejected_wall_indices'].nbytes
        for f in all_frames
    ) / 1e6

    print(f"\n  Total: {total_frames} frames | {mem_mb:.0f} MB RAM | {t_total:.1f}s")
    return all_frames, seq_counts


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

_GLOBAL_FRAMES = None


def _init_worker(frames):
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


def _eval_full_combo(args):
    """Worker: evalúa un combo completo (delta-r + DBSCAN)."""
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


# ========================================
# BÚSQUEDAS PARALELAS
# ========================================

def search_delta_r_parallel(frames, grid, n_workers):
    combos = list(product(grid['threshold_obs'], grid['threshold_void']))
    print(f"  [Delta-r] {len(combos)} combos × {len(frames)} frames, {n_workers} workers")

    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(frames,)) as pool:
        results = pool.map(_eval_delta_r_combo, combos,
                           chunksize=max(1, len(combos) // n_workers))

    t_total = time.time() - t0
    print(f"  [Delta-r] {len(combos)} combos en {t_total:.1f}s")
    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def search_dbscan_parallel(frames, delta_r_params, dbscan_grid, n_workers):
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
            _eval_full_combo, combos,
            chunksize=max(1, len(combos) // (n_workers * 4))
        ))

    t_total = time.time() - t0
    print(f"  [DBSCAN] {len(combos)} combos en {t_total:.1f}s ({t_total/len(combos):.2f}s/combo)")
    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def search_full_parallel(frames, delta_r_grid, dbscan_grid, n_workers):
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
# EVALUACIÓN EN VALIDACIÓN
# ========================================

def evaluate_on_val(val_frames, params_list, mode, n_workers):
    """Evalúa una lista de configuraciones en frames de validación."""
    if not val_frames:
        return []

    if mode == 'delta_r':
        combos = [(p['threshold_obs'], p['threshold_void']) for p in params_list]
        with Pool(n_workers, initializer=_init_worker, initargs=(val_frames,)) as pool:
            val_results = pool.map(_eval_delta_r_combo, combos,
                                   chunksize=max(1, len(combos) // max(1, n_workers)))
    else:
        combos = [
            (p['threshold_obs'], p['threshold_void'],
             p['cluster_eps'], p['cluster_min_samples'], p['cluster_min_pts'])
            for p in params_list
        ]
        with Pool(n_workers, initializer=_init_worker, initargs=(val_frames,)) as pool:
            val_results = pool.map(_eval_full_combo, combos,
                                   chunksize=max(1, len(combos) // max(1, n_workers)))

    val_results.sort(key=lambda x: x['f1'], reverse=True)
    return val_results


# ========================================
# CÁLCULO DE BASELINES
# ========================================

def compute_baselines(frames, current_thr_obs, current_thr_void,
                      current_eps, current_min_samples, current_min_pts, label=""):
    """Calcula baselines sin DBSCAN y con DBSCAN para un set de frames."""
    # Sin DBSCAN
    bl_tp, bl_fp, bl_fn = 0, 0, 0
    obs_masks = []
    for f in frames:
        obs_mask = replay_delta_r_single(f, current_thr_obs, current_thr_void)
        obs_masks.append(obs_mask)
        tp, fp, fn = compute_metrics_accum(f['gt_mask'], obs_mask)
        bl_tp += tp
        bl_fp += fp
        bl_fn += fn
    baseline = metrics_from_accum(bl_tp, bl_fp, bl_fn)

    # Con DBSCAN
    bld_tp, bld_fp, bld_fn = 0, 0, 0
    for i, f in enumerate(frames):
        obs_filtered = replay_dbscan(
            f['points'], obs_masks[i],
            current_eps, current_min_samples, current_min_pts
        )
        tp, fp, fn = compute_metrics_accum(f['gt_mask'], obs_filtered)
        bld_tp += tp
        bld_fp += fp
        bld_fn += fn
    baseline_dbscan = metrics_from_accum(bld_tp, bld_fp, bld_fn)

    if label:
        print(f"\n  {label}:")
    print(f"    Sin DBSCAN: F1={100*baseline['f1']:.2f}%  IoU={100*baseline['iou']:.2f}%  "
          f"P={100*baseline['precision']:.2f}%  R={100*baseline['recall']:.2f}%")
    print(f"    Con DBSCAN: F1={100*baseline_dbscan['f1']:.2f}%  IoU={100*baseline_dbscan['iou']:.2f}%  "
          f"P={100*baseline_dbscan['precision']:.2f}%  R={100*baseline_dbscan['recall']:.2f}%")

    return baseline, baseline_dbscan


# ========================================
# IMPRESIÓN DE RESULTADOS
# ========================================

def print_results(results, baseline_f1, baseline_iou, top_n, title, mode):
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


def print_train_val_comparison(train_results, val_results, baseline_train_f1,
                                baseline_val_f1, top_n, mode):
    """Tabla comparativa train vs val para detectar overfitting."""
    print(f"\n{'='*120}")
    print(f"COMPARACIÓN TRAIN vs VAL (Top {top_n})")
    print(f"Baseline — Train: F1={100*baseline_train_f1:.2f}%  Val: F1={100*baseline_val_f1:.2f}%")
    print(f"{'='*120}")

    # Indexar val por params
    val_by_params = {str(r['params']): r for r in val_results}

    if mode == 'delta_r':
        print(f"{'#':>4} {'thr_obs':>8} {'thr_void':>9}"
              f" | {'Train F1':>9} {'Val F1':>9} {'Gap':>7}"
              f" | {'Train IoU':>10} {'Val IoU':>10}")
        print("-" * 95)
    else:
        print(f"{'#':>4} {'thr_obs':>8} {'thr_void':>9} {'eps':>6} {'ms':>4} {'mp':>4}"
              f" | {'Train F1':>9} {'Val F1':>9} {'Gap':>7}"
              f" | {'Train IoU':>10} {'Val IoU':>10}")
        print("-" * 115)

    for i, tr in enumerate(train_results[:top_n]):
        p = tr['params']
        pkey = str(p)
        vr = val_by_params.get(pkey, None)

        gap = (tr['f1'] - vr['f1']) if vr else float('nan')
        val_f1_str = f"{100*vr['f1']:>8.2f}%" if vr else "    N/A  "
        val_iou_str = f"{100*vr['iou']:>9.2f}%" if vr else "     N/A  "
        gap_str = f"{100*gap:>+6.2f}%" if vr else "   N/A "

        # Marcar si gap > 1% (posible overfitting)
        marker = " !" if vr and gap > 0.01 else ""

        if mode == 'delta_r':
            print(f"{i+1:>4} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                  f" | {100*tr['f1']:>8.2f}% {val_f1_str} {gap_str}"
                  f" | {100*tr['iou']:>9.2f}% {val_iou_str}{marker}")
        else:
            print(f"{i+1:>4} {p['threshold_obs']:>8.2f} {p['threshold_void']:>9.2f}"
                  f" {p['cluster_eps']:>6.2f} {p['cluster_min_samples']:>4} {p['cluster_min_pts']:>4}"
                  f" | {100*tr['f1']:>8.2f}% {val_f1_str} {gap_str}"
                  f" | {100*tr['iou']:>9.2f}% {val_iou_str}{marker}")

    # Mejor por val F1
    if val_results:
        best_val = val_results[0]
        p = best_val['params']
        params_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"\nMEJOR POR VAL F1: {params_str}")
        print(f"  Val:   F1={100*best_val['f1']:.2f}%  IoU={100*best_val['iou']:.2f}%  "
              f"P={100*best_val['precision']:.2f}%  R={100*best_val['recall']:.2f}%")
        # Buscar su train
        tr_match = val_by_params.get(str(best_val['params']))
        train_by_params = {str(r['params']): r for r in train_results}
        tr_r = train_by_params.get(str(best_val['params']))
        if tr_r:
            print(f"  Train: F1={100*tr_r['f1']:.2f}%  IoU={100*tr_r['iou']:.2f}%  "
                  f"P={100*tr_r['precision']:.2f}%  R={100*tr_r['recall']:.2f}%")


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search PARALELO delta-r + DBSCAN con split train/val SemanticKITTI')
    parser.add_argument('--train_seq', type=str, nargs='*', default=None,
                        help='Secuencias train (default: 00-07,09-10 disponibles)')
    parser.add_argument('--val_seq', type=str, nargs='*', default=None,
                        help='Secuencias validación (default: 08)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Usar 1 de cada N frames (default: 1 = todos)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['delta_r', 'dbscan', 'full'])
    parser.add_argument('--workers', type=int, default=0,
                        help='Número de workers (default: cpu_count)')
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--eval_top', type=int, default=50,
                        help='Evaluar las N mejores de train en val (default: 50)')
    parser.add_argument('--threshold_obs', type=float, default=-0.5,
                        help='threshold_obs fijo para modo dbscan (default: -0.5)')
    parser.add_argument('--threshold_void', type=float, default=0.8,
                        help='threshold_void fijo para modo dbscan (default: 0.8)')
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else cpu_count()

    # Descubrir secuencias disponibles
    available = discover_sequences()
    if not available:
        print("ERROR: No se encontraron secuencias con velodyne + labels")
        return

    # Determinar train y val
    if args.train_seq:
        train_seqs = [s for s in args.train_seq if s in available]
    else:
        train_seqs = [s for s in SEMANTICKITTI_TRAIN if s in available]

    if args.val_seq:
        val_seqs = [s for s in args.val_seq if s in available]
    else:
        val_seqs = [s for s in SEMANTICKITTI_VAL if s in available]

    if not train_seqs:
        print("ERROR: No hay secuencias de train disponibles")
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
    print("GRID SEARCH PARALELO — DELTA-R + DBSCAN")
    print("Protocolo SemanticKITTI: tuning en train, evaluación en val")
    print("=" * 120)
    print(f"\nSecuencias disponibles: {available}")
    print(f"Train: {train_seqs}")
    print(f"Val:   {val_seqs if val_seqs else 'NINGUNA (sin evaluación de generalización)'}")
    print(f"Stride: {args.stride}")
    print(f"Workers: {n_workers}")
    print(f"Modo: {args.mode}")
    if args.mode in ('delta_r', 'full'):
        print(f"Delta-r: {n_dr} combos — {delta_r_grid}")
    if args.mode in ('dbscan', 'full'):
        print(f"DBSCAN: {n_db} combos — {dbscan_grid}")
    if args.mode == 'full':
        print(f"Total combos: {n_dr * n_db}")
    print(f"Eval top: {args.eval_top} mejores de train se evalúan en val")

    t0_global = time.time()

    # ========================================
    # FASE 1: PRECOMPUTAR
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 1: PRECOMPUTACIÓN Stage 1 + delta_r")
    print(f"{'='*120}")

    print(f"\n--- TRAIN ({len(train_seqs)} seqs) ---")
    train_frames, train_seq_counts = precompute_all_sequences(train_seqs, args.stride)

    val_frames = []
    val_seq_counts = {}
    if val_seqs:
        print(f"\n--- VAL ({len(val_seqs)} seqs) ---")
        val_frames, val_seq_counts = precompute_all_sequences(val_seqs, args.stride)

    if not train_frames:
        print("ERROR: No hay frames de train")
        return

    # ========================================
    # FASE 2: BASELINES
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 2: BASELINES")
    print(f"{'='*120}")

    current_thr_obs = args.threshold_obs
    current_thr_void = args.threshold_void
    current_eps = 0.8
    current_min_samples = 8
    current_min_pts = 30

    bl_train, bl_train_db = compute_baselines(
        train_frames, current_thr_obs, current_thr_void,
        current_eps, current_min_samples, current_min_pts,
        f"TRAIN ({sum(train_seq_counts.values())} frames)")

    bl_val, bl_val_db = None, None
    if val_frames:
        bl_val, bl_val_db = compute_baselines(
            val_frames, current_thr_obs, current_thr_void,
            current_eps, current_min_samples, current_min_pts,
            f"VAL ({sum(val_seq_counts.values())} frames)")

    # ========================================
    # FASE 3: GRID SEARCH (solo TRAIN)
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 3: GRID SEARCH (solo TRAIN)")
    print(f"{'='*120}")

    if args.mode == 'delta_r':
        train_results = search_delta_r_parallel(train_frames, delta_r_grid, n_workers)
        ref_bl = bl_train
        print_results(train_results, ref_bl['f1'], ref_bl['iou'], args.top,
                      f"TOP {args.top} DELTA-R — TRAIN "
                      f"({sum(train_seq_counts.values())} frames, {len(train_seq_counts)} seqs)",
                      'delta_r')

    elif args.mode == 'dbscan':
        train_results = search_dbscan_parallel(
            train_frames,
            {'threshold_obs': current_thr_obs, 'threshold_void': current_thr_void},
            dbscan_grid, n_workers
        )
        ref_bl = bl_train
        print_results(train_results, ref_bl['f1'], ref_bl['iou'], args.top,
                      f"TOP {args.top} DBSCAN — TRAIN", 'dbscan')

    elif args.mode == 'full':
        train_results = search_full_parallel(train_frames, delta_r_grid, dbscan_grid, n_workers)
        ref_bl = bl_train_db
        print_results(train_results, ref_bl['f1'], ref_bl['iou'], args.top,
                      f"TOP {args.top} FULL — TRAIN "
                      f"({sum(train_seq_counts.values())} frames, {len(train_seq_counts)} seqs)",
                      'full')

    # ========================================
    # FASE 4: EVALUACIÓN EN VAL
    # ========================================
    val_results = []
    if val_frames and train_results:
        print(f"\n{'='*120}")
        print(f"FASE 4: EVALUACIÓN EN VAL (seq {val_seqs}, {sum(val_seq_counts.values())} frames)")
        print(f"{'='*120}")

        # Evaluar las top-N de train en val
        top_params = [r['params'] for r in train_results[:args.eval_top]]
        print(f"\n  Evaluando {len(top_params)} mejores configs de train en val...", end=" ", flush=True)
        t0 = time.time()

        val_results = evaluate_on_val(val_frames, top_params, args.mode, n_workers)
        print(f"OK ({time.time()-t0:.1f}s)")

        # Resultados val
        val_ref_f1 = bl_val_db['f1'] if args.mode == 'full' else bl_val['f1']
        val_ref_iou = bl_val_db['iou'] if args.mode == 'full' else bl_val['iou']
        print_results(val_results, val_ref_f1, val_ref_iou, args.top,
                      f"TOP {args.top} — VAL (seq {val_seqs})", args.mode)

        # Comparación train vs val
        train_ref_f1 = ref_bl['f1']
        print_train_val_comparison(
            train_results, val_results, train_ref_f1, val_ref_f1, args.top, args.mode
        )

    # ========================================
    # RESUMEN FINAL
    # ========================================
    t_total = time.time() - t0_global
    print(f"\n{'='*120}")
    print("RESUMEN FINAL")
    print(f"{'='*120}")
    print(f"  Train seqs: {list(train_seq_counts.keys())} ({sum(train_seq_counts.values())} frames)")
    if val_seq_counts:
        print(f"  Val seqs:   {list(val_seq_counts.keys())} ({sum(val_seq_counts.values())} frames)")
    print(f"\n  Baselines:")
    print(f"    Train sin DBSCAN: F1={100*bl_train['f1']:.2f}%  IoU={100*bl_train['iou']:.2f}%")
    print(f"    Train con DBSCAN: F1={100*bl_train_db['f1']:.2f}%  IoU={100*bl_train_db['iou']:.2f}%")
    if bl_val:
        print(f"    Val sin DBSCAN:   F1={100*bl_val['f1']:.2f}%  IoU={100*bl_val['iou']:.2f}%")
        print(f"    Val con DBSCAN:   F1={100*bl_val_db['f1']:.2f}%  IoU={100*bl_val_db['iou']:.2f}%")

    if train_results:
        best_train = train_results[0]
        p = best_train['params']
        params_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"\n  Mejor en TRAIN:")
        print(f"    {params_str}")
        print(f"    F1={100*best_train['f1']:.2f}%  IoU={100*best_train['iou']:.2f}%")

    if val_results:
        best_val = val_results[0]
        p = best_val['params']
        params_str = ', '.join(f"{k}={v}" for k, v in p.items())
        print(f"\n  Mejor en VAL (resultado reportable):")
        print(f"    {params_str}")
        print(f"    F1={100*best_val['f1']:.2f}%  IoU={100*best_val['iou']:.2f}%  "
              f"P={100*best_val['precision']:.2f}%  R={100*best_val['recall']:.2f}%")

    print(f"\n  Tiempo total: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == '__main__':
    main()
