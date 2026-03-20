#!/usr/bin/env python3
"""
Grid Search PARALELO de parámetros de Wall Rejection (Stage 1).

Protocolo SemanticKITTI reproducible:
  - TRAIN (tuning):  seq 00-07, 09-10
  - VAL (evaluación): seq 08

Diseñado para ejecutar DESPUÉS del grid search de delta-r + DBSCAN.
Fija los parámetros óptimos de Stage 2-3 y optimiza Stage 1.

Parámetros explorados:
  - wall_rejection_slope:       Umbral nz de la normal (< umbral → pared)
  - wall_height_diff_threshold: Delta-Z mínimo para confirmar pared (m)
  - wall_kdtree_radius:         Radio de vecindad local para delta-Z (m)

NOTA: Cada combo requiere re-ejecutar Stage 1 completo (Patchwork++ + wall rejection
+ delta-r), más lento que el grid search de delta-r/DBSCAN.

Uso:
    # Con parámetros óptimos de delta-r + DBSCAN
    python3 tests/test_wall_rejection_grid_search.py \\
        --threshold_obs -0.6 --threshold_void 1.2 \\
        --cluster_eps 1.0 --cluster_min_samples 12 --cluster_min_pts 50 \\
        --workers 128 --stride 5

    # Sin DBSCAN (solo Stage 1 + 2)
    python3 tests/test_wall_rejection_grid_search.py --no_dbscan --stride 5
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir,
                        get_labels_dir, VELODYNE_ROOT, LABELS_ROOT)

SEMANTICKITTI_TRAIN = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
SEMANTICKITTI_VAL = ['08']


# ========================================
# AUTO-DESCUBRIMIENTO
# ========================================

def discover_sequences():
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

OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    50, 51, 52, 70, 71, 80, 81, 99,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)


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


def get_gt_obstacle_mask(semantic_labels):
    return np.isin(semantic_labels, OBSTACLE_LABELS)


def compute_metrics_accum(gt_mask, pred_mask):
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    return tp, fp, fn


def metrics_from_accum(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn}


# ========================================
# GRID
# ========================================

def get_wall_rejection_grid():
    return {
        'wall_rejection_slope':       [0.5, 0.6, 0.7, 0.8, 0.9],
        'wall_height_diff_threshold': [0.15, 0.2, 0.3, 0.4, 0.5],
        'wall_kdtree_radius':         [0.3, 0.5, 0.7, 1.0],
    }


# ========================================
# REPLAY DBSCAN
# ========================================

def replay_dbscan(points, obs_mask, eps, min_samples, min_pts):
    from sklearn.cluster import DBSCAN as _DBSCAN

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

    db = _DBSCAN(eps=eps, min_samples=max(2, min_samples // 2), n_jobs=1)
    voxel_labels = db.fit_predict(voxel_centroids)
    cluster_labels_obs = voxel_labels[inverse]

    max_label = cluster_labels_obs.max()
    N = len(points)
    if max_label >= 0:
        cluster_sizes = np.bincount(cluster_labels_obs[cluster_labels_obs >= 0],
                                     minlength=max_label + 1)
        point_cluster_size = np.where(
            cluster_labels_obs >= 0,
            cluster_sizes[cluster_labels_obs.clip(0)], 0)
        valid_mask_obs = point_cluster_size >= min_pts
    else:
        valid_mask_obs = np.zeros(n_obs, dtype=bool)

    obs_mask_new = np.zeros(N, dtype=bool)
    obs_mask_new[obs_indices[valid_mask_obs]] = True
    return obs_mask_new


# ========================================
# CARGA DE DATOS
# ========================================

def load_all_data(seqs, stride):
    """Carga puntos y GT para todas las secuencias."""
    all_data = []
    seq_counts = {}
    t0 = time.time()

    for seq in seqs:
        scan_ids = discover_scan_ids(seq, stride=stride)
        if not scan_ids:
            print(f"  Seq {seq}: SKIP")
            continue

        print(f"  Seq {seq}: {len(scan_ids)} frames...", end=" ", flush=True)
        t_seq = time.time()

        for scan_id in scan_ids:
            pts, labels = load_kitti_scan(scan_id, seq)
            gt_mask = get_gt_obstacle_mask(labels)
            all_data.append((seq, scan_id, pts, gt_mask))

        seq_counts[seq] = len(scan_ids)
        print(f"OK ({time.time()-t_seq:.1f}s)")

    t_total = time.time() - t0
    mem_mb = sum(d[2].nbytes + d[3].nbytes for d in all_data) / 1e6
    print(f"\n  Total: {len(all_data)} frames | {mem_mb:.0f} MB | {t_total:.1f}s")
    return all_data, seq_counts


# ========================================
# MULTIPROCESSING
# ========================================

_GLOBAL_DATA = None
_GLOBAL_FIXED_PARAMS = None


def _init_worker(data, fixed_params):
    global _GLOBAL_DATA, _GLOBAL_FIXED_PARAMS
    _GLOBAL_DATA = data
    _GLOBAL_FIXED_PARAMS = fixed_params


def _eval_wall_combo(args):
    """Worker: ejecuta pipeline S1+S2(+S3) con un combo de wall rejection."""
    slope, dz_thresh, kd_radius = args
    fp = _GLOBAL_FIXED_PARAMS

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=slope,
        wall_height_diff_threshold=dz_thresh,
        wall_kdtree_radius=kd_radius,
        threshold_obs=fp['threshold_obs'],
        threshold_void=fp['threshold_void'],
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    total_tp, total_fp, total_fn = 0, 0, 0
    total_walls = 0

    for seq, scan_id, pts, gt_mask in _GLOBAL_DATA:
        result = pipeline.stage2_complete(pts)
        obs_mask = result['obs_mask']
        total_walls += len(result.get('rejected_walls', []))

        if fp['use_dbscan']:
            obs_mask = replay_dbscan(
                pts, obs_mask,
                fp['cluster_eps'], fp['cluster_min_samples'], fp['cluster_min_pts']
            )

        tp, fp_val, fn = compute_metrics_accum(gt_mask, obs_mask)
        total_tp += tp
        total_fp += fp_val
        total_fn += fn

    m = metrics_from_accum(total_tp, total_fp, total_fn)
    return {
        'params': {
            'wall_rejection_slope': slope,
            'wall_height_diff_threshold': dz_thresh,
            'wall_kdtree_radius': kd_radius,
        },
        'total_walls': total_walls,
        **m,
    }


def _eval_no_wall_baseline(dummy):
    """Worker: pipeline sin wall rejection."""
    fp = _GLOBAL_FIXED_PARAMS

    config = PipelineConfig(
        enable_hybrid_wall_rejection=False,
        threshold_obs=fp['threshold_obs'],
        threshold_void=fp['threshold_void'],
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    total_tp, total_fp, total_fn = 0, 0, 0
    for seq, scan_id, pts, gt_mask in _GLOBAL_DATA:
        result = pipeline.stage2_complete(pts)
        obs_mask = result['obs_mask']

        if fp['use_dbscan']:
            obs_mask = replay_dbscan(
                pts, obs_mask,
                fp['cluster_eps'], fp['cluster_min_samples'], fp['cluster_min_pts']
            )

        tp, fp_val, fn = compute_metrics_accum(gt_mask, obs_mask)
        total_tp += tp
        total_fp += fp_val
        total_fn += fn

    return metrics_from_accum(total_tp, total_fp, total_fn)


def run_grid_search(data, fixed_params, n_workers):
    """Ejecuta grid search paralelo sobre datos."""
    grid = get_wall_rejection_grid()
    combos = list(product(
        grid['wall_rejection_slope'],
        grid['wall_height_diff_threshold'],
        grid['wall_kdtree_radius'],
    ))

    print(f"  {len(combos)} combos × {len(data)} frames, {n_workers} workers")

    t0 = time.time()
    done = [0]

    with Pool(n_workers, initializer=_init_worker, initargs=(data, fixed_params)) as pool:
        results = []
        chunksize = max(1, len(combos) // (n_workers * 2))
        for r in pool.imap_unordered(_eval_wall_combo, combos, chunksize=chunksize):
            results.append(r)
            done[0] += 1
            if done[0] % max(1, len(combos) // 10) == 0 or done[0] == len(combos):
                elapsed = time.time() - t0
                eta = (elapsed / done[0]) * (len(combos) - done[0])
                print(f"\r  [{done[0]}/{len(combos)}] ({100*done[0]/len(combos):.0f}%) "
                      f"ETA: {eta:.0f}s  ", end="", flush=True)

    t_total = time.time() - t0
    print(f"\r  {len(combos)} combos en {t_total:.1f}s "
          f"({t_total/len(combos):.1f}s/combo)" + " " * 30)

    results.sort(key=lambda x: x['f1'], reverse=True)
    return results


def compute_baselines(data, fixed_params, n_workers, label=""):
    """Calcula baseline sin wall rejection y con parámetros actuales."""
    # Sin wall rejection
    print(f"  {label} — Sin wall rejection...", end=" ", flush=True)
    t0 = time.time()
    with Pool(1, initializer=_init_worker, initargs=(data, fixed_params)) as pool:
        bl_no_wall = pool.apply(_eval_no_wall_baseline, (None,))
    print(f"OK ({time.time()-t0:.1f}s)")
    print(f"    F1={100*bl_no_wall['f1']:.2f}%  IoU={100*bl_no_wall['iou']:.2f}%  "
          f"P={100*bl_no_wall['precision']:.2f}%  R={100*bl_no_wall['recall']:.2f}%")

    # Con wall rejection actual (0.7, 0.3, 0.5)
    print(f"  {label} — Con wall rejection (0.7/0.3/0.5)...", end=" ", flush=True)
    t0 = time.time()
    with Pool(1, initializer=_init_worker, initargs=(data, fixed_params)) as pool:
        bl_current = pool.apply(_eval_wall_combo, ((0.7, 0.3, 0.5),))
    print(f"OK ({time.time()-t0:.1f}s)")
    print(f"    F1={100*bl_current['f1']:.2f}%  IoU={100*bl_current['iou']:.2f}%  "
          f"P={100*bl_current['precision']:.2f}%  R={100*bl_current['recall']:.2f}%  "
          f"Walls={bl_current['total_walls']}")

    return bl_no_wall, bl_current


def print_results(results, baseline_f1, top_n, title):
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"Baseline (actual): F1={100*baseline_f1:.2f}%")
    print(f"{'='*120}")

    print(f"{'#':>4} {'slope':>6} {'dz_thr':>7} {'kd_r':>6}"
          f" | {'F1':>8} {'IoU':>8} {'dF1':>8} {'P':>8} {'R':>8}"
          f" {'FP':>10} {'FN':>10} {'Walls':>8}")
    print("-" * 115)

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        df1 = r['f1'] - baseline_f1
        marker = " *" if df1 > 0.0001 else ""
        print(f"{i+1:>4} {p['wall_rejection_slope']:>6.2f} {p['wall_height_diff_threshold']:>7.2f}"
              f" {p['wall_kdtree_radius']:>6.2f}"
              f" | {100*r['f1']:>7.2f}% {100*r['iou']:>7.2f}% {100*df1:>+7.2f}%"
              f" {100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}%"
              f" {r['fp']:>10} {r['fn']:>10} {r['total_walls']:>8}{marker}")

    n_better = sum(1 for r in results if r['f1'] > baseline_f1 + 0.0001)
    print(f"\n{n_better}/{len(results)} mejoran vs baseline actual")


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search PARALELO — Wall Rejection (Stage 1) con split train/val')

    parser.add_argument('--train_seq', type=str, nargs='*', default=None)
    parser.add_argument('--val_seq', type=str, nargs='*', default=None)
    parser.add_argument('--stride', type=int, default=1)

    parser.add_argument('--threshold_obs', type=float, default=-0.5)
    parser.add_argument('--threshold_void', type=float, default=0.8)
    parser.add_argument('--no_dbscan', action='store_true')
    parser.add_argument('--cluster_eps', type=float, default=0.8)
    parser.add_argument('--cluster_min_samples', type=int, default=8)
    parser.add_argument('--cluster_min_pts', type=int, default=30)

    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--eval_top', type=int, default=30,
                        help='Evaluar las N mejores de train en val')

    args = parser.parse_args()
    n_workers = args.workers if args.workers > 0 else cpu_count()

    available = discover_sequences()
    if not available:
        print("ERROR: No se encontraron secuencias")
        return

    train_seqs = args.train_seq if args.train_seq else [s for s in SEMANTICKITTI_TRAIN if s in available]
    val_seqs = args.val_seq if args.val_seq else [s for s in SEMANTICKITTI_VAL if s in available]

    if not train_seqs:
        print("ERROR: No hay secuencias de train")
        return

    fixed_params = {
        'threshold_obs': args.threshold_obs,
        'threshold_void': args.threshold_void,
        'use_dbscan': not args.no_dbscan,
        'cluster_eps': args.cluster_eps,
        'cluster_min_samples': args.cluster_min_samples,
        'cluster_min_pts': args.cluster_min_pts,
    }

    grid = get_wall_rejection_grid()
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print("=" * 120)
    print("GRID SEARCH PARALELO — WALL REJECTION (Stage 1)")
    print("Protocolo SemanticKITTI: tuning en train, evaluación en val")
    print("=" * 120)
    print(f"\nTrain: {train_seqs}")
    print(f"Val:   {val_seqs if val_seqs else 'NINGUNA'}")
    print(f"Stride: {args.stride} | Workers: {n_workers} | Combos: {n_combos}")
    print(f"\nGrid: {grid}")
    print(f"\nStage 2-3 fijos: thr_obs={args.threshold_obs}, thr_void={args.threshold_void}")
    if not args.no_dbscan:
        print(f"  DBSCAN: eps={args.cluster_eps}, ms={args.cluster_min_samples}, mp={args.cluster_min_pts}")
    else:
        print(f"  DBSCAN: desactivado")

    t0_global = time.time()

    # ========================================
    # CARGA
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 1: CARGA DE DATOS")
    print(f"{'='*120}")

    print(f"\n--- TRAIN ---")
    train_data, train_counts = load_all_data(train_seqs, args.stride)

    val_data, val_counts = [], {}
    if val_seqs:
        print(f"\n--- VAL ---")
        val_data, val_counts = load_all_data(val_seqs, args.stride)

    if not train_data:
        print("ERROR: No hay datos de train")
        return

    # ========================================
    # BASELINES
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 2: BASELINES")
    print(f"{'='*120}")

    bl_train_no_wall, bl_train_current = compute_baselines(
        train_data, fixed_params, n_workers, "TRAIN")

    bl_val_no_wall, bl_val_current = None, None
    if val_data:
        bl_val_no_wall, bl_val_current = compute_baselines(
            val_data, fixed_params, n_workers, "VAL")

    # ========================================
    # GRID SEARCH (TRAIN)
    # ========================================
    print(f"\n{'='*120}")
    print("FASE 3: GRID SEARCH (solo TRAIN)")
    print(f"{'='*120}\n")

    train_results = run_grid_search(train_data, fixed_params, n_workers)
    print_results(train_results, bl_train_current['f1'], args.top,
                  f"TOP {args.top} WALL REJECTION — TRAIN "
                  f"({sum(train_counts.values())} frames, {len(train_counts)} seqs)")

    # ========================================
    # EVALUACIÓN EN VAL
    # ========================================
    val_results = []
    if val_data and train_results:
        print(f"\n{'='*120}")
        print(f"FASE 4: EVALUACIÓN EN VAL (seq {val_seqs}, {sum(val_counts.values())} frames)")
        print(f"{'='*120}\n")

        # Evaluar top-N de train en val
        top_combos = [
            (r['params']['wall_rejection_slope'],
             r['params']['wall_height_diff_threshold'],
             r['params']['wall_kdtree_radius'])
            for r in train_results[:args.eval_top]
        ]

        print(f"  Evaluando {len(top_combos)} mejores de train en val...", end=" ", flush=True)
        t0 = time.time()
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(val_data, fixed_params)) as pool:
            val_results = list(pool.imap_unordered(
                _eval_wall_combo, top_combos,
                chunksize=max(1, len(top_combos) // max(1, n_workers))
            ))
        val_results.sort(key=lambda x: x['f1'], reverse=True)
        print(f"OK ({time.time()-t0:.1f}s)")

        print_results(val_results, bl_val_current['f1'], args.top,
                      f"TOP {args.top} WALL REJECTION — VAL (seq {val_seqs})")

        # Comparación train vs val
        print(f"\n{'='*120}")
        print(f"COMPARACIÓN TRAIN vs VAL (Top {args.top})")
        print(f"{'='*120}")

        val_by_params = {str(r['params']): r for r in val_results}
        print(f"{'#':>4} {'slope':>6} {'dz':>5} {'r':>5}"
              f" | {'Train F1':>9} {'Val F1':>9} {'Gap':>7}"
              f" | {'Train IoU':>10} {'Val IoU':>10}")
        print("-" * 85)

        for i, tr in enumerate(train_results[:args.top]):
            p = tr['params']
            vr = val_by_params.get(str(p))
            gap = (tr['f1'] - vr['f1']) if vr else float('nan')
            val_f1 = f"{100*vr['f1']:>8.2f}%" if vr else "    N/A  "
            val_iou = f"{100*vr['iou']:>9.2f}%" if vr else "     N/A  "
            gap_str = f"{100*gap:>+6.2f}%" if vr else "   N/A "
            marker = " !" if vr and gap > 0.01 else ""
            print(f"{i+1:>4} {p['wall_rejection_slope']:>6.2f} {p['wall_height_diff_threshold']:>5.2f}"
                  f" {p['wall_kdtree_radius']:>5.2f}"
                  f" | {100*tr['f1']:>8.2f}% {val_f1} {gap_str}"
                  f" | {100*tr['iou']:>9.2f}% {val_iou}{marker}")

    # ========================================
    # RESUMEN
    # ========================================
    t_total = time.time() - t0_global
    print(f"\n{'='*120}")
    print("RESUMEN FINAL")
    print(f"{'='*120}")
    print(f"  Train: {list(train_counts.keys())} ({sum(train_counts.values())} frames)")
    if val_counts:
        print(f"  Val:   {list(val_counts.keys())} ({sum(val_counts.values())} frames)")

    print(f"\n  Baselines TRAIN:")
    print(f"    Sin wall rejection: F1={100*bl_train_no_wall['f1']:.2f}%")
    print(f"    Actual (0.7/0.3/0.5): F1={100*bl_train_current['f1']:.2f}%")

    if bl_val_current:
        print(f"  Baselines VAL:")
        print(f"    Sin wall rejection: F1={100*bl_val_no_wall['f1']:.2f}%")
        print(f"    Actual (0.7/0.3/0.5): F1={100*bl_val_current['f1']:.2f}%")

    if train_results:
        best = train_results[0]
        p = best['params']
        print(f"\n  Mejor en TRAIN: slope={p['wall_rejection_slope']}, "
              f"dz={p['wall_height_diff_threshold']}, r={p['wall_kdtree_radius']}")
        print(f"    F1={100*best['f1']:.2f}%  Walls={best['total_walls']}")

    if val_results:
        best_val = val_results[0]
        p = best_val['params']
        print(f"\n  Mejor en VAL (reportable): slope={p['wall_rejection_slope']}, "
              f"dz={p['wall_height_diff_threshold']}, r={p['wall_kdtree_radius']}")
        print(f"    F1={100*best_val['f1']:.2f}%  IoU={100*best_val['iou']:.2f}%  "
              f"P={100*best_val['precision']:.2f}%  R={100*best_val['recall']:.2f}%")

    print(f"\n  Tiempo total: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == '__main__':
    main()
