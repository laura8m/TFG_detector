#!/usr/bin/env python3
"""
Grid Search PARALELO de parámetros de Curb Detection (Feature 1).

Optimiza los parámetros del detector de bordillos por discontinuidad inter-ring
y compara ablación con/sin curb detection.

Parámetros explorados:
  - curb_height_min:      Altura mínima del bordillo (m)
  - curb_height_max:      Altura máxima (>max = pared)
  - curb_min_consecutive: Puntos consecutivos mínimos para confirmar bordillo

Protocolo SemanticKITTI reproducible:
  - TRAIN (tuning):  seq 00-07, 09-10
  - VAL (evaluación): seq 08

Uso:
    python3 tests/test_curb_detection_grid_search.py --stride 5 --workers 64
    python3 tests/test_curb_detection_grid_search.py --seq 00 04 --stride 10  # rápido
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
import json
from itertools import product
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir,
                        get_labels_dir, VELODYNE_ROOT, LABELS_ROOT,
                        OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL)

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


def get_gt_curb_mask(semantic_labels):
    """Máscara de puntos que son bordillo (curb=3) en ground truth."""
    return semantic_labels == CURB_LABEL


def compute_metrics_accum(gt_mask, pred_mask, valid_mask=None):
    if valid_mask is not None:
        gt_mask = gt_mask & valid_mask
        pred_mask = pred_mask & valid_mask
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

def get_curb_grid():
    return {
        'curb_height_min':      [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
        'curb_height_max':      [0.20, 0.25, 0.30, 0.35, 0.40],
        'curb_min_consecutive': [1, 2, 3, 5, 8],
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
            gt_obs_mask = get_gt_obstacle_mask(labels)
            gt_curb_mask = get_gt_curb_mask(labels)
            valid_mask = ~np.isin(labels, IGNORE_LABELS)
            all_data.append((seq, scan_id, pts, gt_obs_mask, gt_curb_mask, valid_mask))

        seq_counts[seq] = len(scan_ids)
        print(f"OK ({time.time()-t_seq:.1f}s)")

    t_total = time.time() - t0
    n_curb_total = sum(d[4].sum() for d in all_data)
    print(f"\n  Total: {len(all_data)} frames | {n_curb_total} curb GT pts | {t_total:.1f}s")
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


def _eval_curb_combo(args):
    """Worker: ejecuta pipeline con un combo de curb detection."""
    h_min, h_max, min_consec = args
    fp = _GLOBAL_FIXED_PARAMS

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=fp['wall_rejection_slope'],
        wall_height_diff_threshold=fp['wall_height_diff_threshold'],
        wall_kdtree_radius=fp['wall_kdtree_radius'],
        enable_curb_detection=True,
        curb_height_min=h_min,
        curb_height_max=h_max,
        curb_min_consecutive=min_consec,
        enable_delta_r=False,
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    # Métricas globales (obstáculos)
    total_tp, total_fp, total_fn = 0, 0, 0
    # Métricas específicas de bordillos
    curb_tp, curb_fn = 0, 0
    total_curb_detected = 0

    for seq, scan_id, pts, gt_obs, gt_curb, valid_mask in _GLOBAL_DATA:
        result = pipeline.stage2_complete(pts)
        obs_mask = result['obs_mask']
        curb_mask = result.get('curb_mask', np.zeros(len(pts), dtype=bool))

        if fp['use_dbscan']:
            obs_mask = replay_dbscan(
                pts, obs_mask,
                fp['cluster_eps'], fp['cluster_min_samples'], fp['cluster_min_pts']
            )

        # Métricas globales
        tp, fp_val, fn = compute_metrics_accum(gt_obs, obs_mask, valid_mask)
        total_tp += tp
        total_fp += fp_val
        total_fn += fn

        # Métricas de bordillos (cuántos GT curb detectamos)
        curb_valid = gt_curb & valid_mask
        curb_tp += int(np.sum(curb_valid & obs_mask))
        curb_fn += int(np.sum(curb_valid & ~obs_mask))
        total_curb_detected += int(curb_mask.sum())

    m = metrics_from_accum(total_tp, total_fp, total_fn)
    curb_recall = curb_tp / (curb_tp + curb_fn) if (curb_tp + curb_fn) > 0 else 0.0
    return {
        'params': {
            'curb_height_min': h_min,
            'curb_height_max': h_max,
            'curb_min_consecutive': min_consec,
        },
        'curb_recall': curb_recall,
        'curb_tp': curb_tp,
        'curb_fn': curb_fn,
        'curb_detected': total_curb_detected,
        **m,
    }


def _eval_no_curb_baseline(dummy):
    """Worker: pipeline sin curb detection (baseline)."""
    fp = _GLOBAL_FIXED_PARAMS

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=fp['wall_rejection_slope'],
        wall_height_diff_threshold=fp['wall_height_diff_threshold'],
        wall_kdtree_radius=fp['wall_kdtree_radius'],
        enable_curb_detection=False,
        enable_delta_r=False,
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    total_tp, total_fp, total_fn = 0, 0, 0
    curb_tp, curb_fn = 0, 0

    for seq, scan_id, pts, gt_obs, gt_curb, valid_mask in _GLOBAL_DATA:
        result = pipeline.stage2_complete(pts)
        obs_mask = result['obs_mask']

        if fp['use_dbscan']:
            obs_mask = replay_dbscan(
                pts, obs_mask,
                fp['cluster_eps'], fp['cluster_min_samples'], fp['cluster_min_pts']
            )

        tp, fp_val, fn = compute_metrics_accum(gt_obs, obs_mask, valid_mask)
        total_tp += tp
        total_fp += fp_val
        total_fn += fn

        curb_valid = gt_curb & valid_mask
        curb_tp += int(np.sum(curb_valid & obs_mask))
        curb_fn += int(np.sum(curb_valid & ~obs_mask))

    m = metrics_from_accum(total_tp, total_fp, total_fn)
    curb_recall = curb_tp / (curb_tp + curb_fn) if (curb_tp + curb_fn) > 0 else 0.0
    return {**m, 'curb_recall': curb_recall, 'curb_tp': curb_tp, 'curb_fn': curb_fn}


# ========================================
# GRID SEARCH
# ========================================

def run_grid_search(data, fixed_params, n_workers):
    grid = get_curb_grid()
    combos = list(product(
        grid['curb_height_min'],
        grid['curb_height_max'],
        grid['curb_min_consecutive'],
    ))

    print(f"  {len(combos)} combos × {len(data)} frames, {n_workers} workers")

    t0 = time.time()
    done = [0]

    with Pool(n_workers, initializer=_init_worker, initargs=(data, fixed_params)) as pool:
        results = []
        chunksize = max(1, len(combos) // (n_workers * 2))
        for r in pool.imap_unordered(_eval_curb_combo, combos, chunksize=chunksize):
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


# ========================================
# RESULTADOS
# ========================================

def print_results(results, baseline, top_n, title):
    print(f"\n{'='*130}")
    print(f"{title}")
    print(f"Baseline (sin curb): F1={100*baseline['f1']:.2f}%  "
          f"Curb Recall={100*baseline['curb_recall']:.2f}%")
    print(f"{'='*130}")

    print(f"{'#':>4} {'h_min':>6} {'h_max':>6} {'consec':>7}"
          f" | {'F1':>8} {'dF1':>8} {'IoU':>8} {'P':>8} {'R':>8}"
          f" | {'CurbR':>8} {'CurbTP':>8} {'CurbFN':>8} {'CurbDet':>8}")
    print("-" * 125)

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        df1 = r['f1'] - baseline['f1']
        print(f"{i+1:4d} {p['curb_height_min']:6.2f} {p['curb_height_max']:6.2f} "
              f"{p['curb_min_consecutive']:7d}"
              f" | {100*r['f1']:7.2f}% {100*df1:+7.2f}% {100*r['iou']:7.2f}% "
              f"{100*r['precision']:7.2f}% {100*r['recall']:7.2f}%"
              f" | {100*r['curb_recall']:7.2f}% {r['curb_tp']:8d} {r['curb_fn']:8d} "
              f"{r['curb_detected']:8d}")


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Grid search curb detection')
    parser.add_argument('--seq', nargs='+', default=None,
                        help='Secuencias a evaluar (default: auto-discover)')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride entre frames (default: 5)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Workers paralelos (0=auto)')
    parser.add_argument('--top', type=int, default=20,
                        help='Top N resultados a mostrar')

    # Parámetros fijos de wall rejection (óptimos)
    parser.add_argument('--wall_slope', type=float, default=0.9)
    parser.add_argument('--wall_dz', type=float, default=0.2)
    parser.add_argument('--wall_radius', type=float, default=0.3)

    # DBSCAN (opcional)
    parser.add_argument('--no_dbscan', action='store_true',
                        help='No aplicar DBSCAN')
    parser.add_argument('--cluster_eps', type=float, default=1.2)
    parser.add_argument('--cluster_min_samples', type=int, default=12)
    parser.add_argument('--cluster_min_pts', type=int, default=10)

    args = parser.parse_args()
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 2)

    # Determinar secuencias
    if args.seq:
        seqs = args.seq
    else:
        available = discover_sequences()
        seqs = [s for s in SEMANTICKITTI_TRAIN if s in available]
        if not seqs:
            seqs = available
    print(f"Secuencias: {seqs}")

    fixed_params = {
        'wall_rejection_slope': args.wall_slope,
        'wall_height_diff_threshold': args.wall_dz,
        'wall_kdtree_radius': args.wall_radius,
        'use_dbscan': not args.no_dbscan,
        'cluster_eps': args.cluster_eps,
        'cluster_min_samples': args.cluster_min_samples,
        'cluster_min_pts': args.cluster_min_pts,
    }

    # Cargar datos
    print(f"\nCargando datos (stride={args.stride})...")
    data, seq_counts = load_all_data(seqs, args.stride)
    if not data:
        print("ERROR: No se encontraron datos.")
        return

    # Baseline: sin curb detection
    print(f"\n[BASELINE] Sin curb detection...")
    t0 = time.time()
    with Pool(1, initializer=_init_worker, initargs=(data, fixed_params)) as pool:
        baseline = pool.apply(_eval_no_curb_baseline, (None,))
    print(f"  F1={100*baseline['f1']:.2f}%  IoU={100*baseline['iou']:.2f}%  "
          f"P={100*baseline['precision']:.2f}%  R={100*baseline['recall']:.2f}%  "
          f"Curb Recall={100*baseline['curb_recall']:.2f}%  "
          f"({time.time()-t0:.1f}s)")

    # Grid search
    print(f"\n[GRID SEARCH] Curb detection...")
    results = run_grid_search(data, fixed_params, n_workers)

    # Mostrar resultados
    print_results(results, baseline, args.top, "CURB DETECTION GRID SEARCH — Ordenado por F1")

    # También ordenar por curb recall
    results_by_curb = sorted(results, key=lambda x: x['curb_recall'], reverse=True)
    print_results(results_by_curb, baseline, args.top,
                  "CURB DETECTION GRID SEARCH — Ordenado por Curb Recall")

    # Mejor combo
    best = results[0]
    print(f"\n{'='*80}")
    print(f"MEJOR COMBO (F1):")
    print(f"  curb_height_min:      {best['params']['curb_height_min']}")
    print(f"  curb_height_max:      {best['params']['curb_height_max']}")
    print(f"  curb_min_consecutive: {best['params']['curb_min_consecutive']}")
    print(f"  F1:         {100*best['f1']:.2f}% (baseline: {100*baseline['f1']:.2f}%, "
          f"delta: {100*(best['f1']-baseline['f1']):+.2f}%)")
    print(f"  Curb Recall: {100*best['curb_recall']:.2f}% "
          f"(baseline: {100*baseline['curb_recall']:.2f}%)")
    print(f"{'='*80}")

    # Guardar resultados
    output = {
        'baseline': baseline,
        'best_f1': {**best},
        'best_curb_recall': {**results_by_curb[0]},
        'fixed_params': fixed_params,
        'sequences': seqs,
        'stride': args.stride,
        'n_frames': len(data),
        'all_results': results[:50],
    }
    out_file = Path(__file__).parent / 'results_curb_grid_search.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResultados guardados en: {out_file}")


if __name__ == "__main__":
    main()
