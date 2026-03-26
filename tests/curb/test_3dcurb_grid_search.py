#!/usr/bin/env python3
"""
Grid Search secuencial para 3D-Curb dataset.

Ejecuta 3 fases de optimización:
  Fase 1: Wall Rejection (slope, dz, radius)
  Fase 2: Feature 1 — bordillos (curb_min, curb_max, consecutive)
  Fase 3: Delta-r conservador (threshold_obs, threshold_void, min_nz)

Cada fase usa los mejores parámetros de la anterior.
Train: seq 00-07, 09-10 | Val: seq 08

Uso:
    python3 tests/curb/test_3dcurb_grid_search.py --workers 128 --stride 5
    python3 tests/curb/test_3dcurb_grid_search.py --workers 128 --stride 5 --phase 2
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))
from data_paths_3dcurb import (
    discover_scan_ids, OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL,
    TRAIN_SEQS, VAL_SEQS, get_velodyne_dir, get_labels_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# Datos globales para multiprocessing
_GLOBAL_TRAIN = None
_GLOBAL_VAL = None


def compute_metrics(gt_mask, pred_mask, valid_mask):
    g = gt_mask & valid_mask
    p = pred_mask & valid_mask
    tp = int(np.sum(g & p))
    fp = int(np.sum(~g & p))
    fn = int(np.sum(g & ~p))
    return tp, fp, fn


def metrics_from_accum(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return f1, iou, p, r


def curb_recall_from_data(curb_gt, pred_mask, valid_mask):
    c = curb_gt & valid_mask
    total = int(np.sum(c))
    detected = int(np.sum(c & pred_mask & valid_mask))
    return detected, total


def load_all_data(seqs, stride, velodyne_root=None, labels_root=None):
    """Carga todos los frames de las secuencias dadas."""
    data = []
    for seq in seqs:
        scan_ids = discover_scan_ids(seq, stride, velodyne_root, labels_root)
        if not scan_ids:
            print(f"  Seq {seq}: sin frames")
            continue
        t0 = time.time()
        print(f"  Seq {seq}: {len(scan_ids)} frames...", end=" ", flush=True)
        vel_dir = get_velodyne_dir(seq, velodyne_root)
        lab_dir = get_labels_dir(seq, labels_root)
        for sid in scan_ids:
            pts = np.fromfile(str(vel_dir / f"{sid:06d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
            lbl = np.fromfile(str(lab_dir / f"{sid:06d}.label"), dtype=np.uint32) & 0xFFFF
            gt = np.isin(lbl, OBSTACLE_LABELS)
            valid = ~np.isin(lbl, IGNORE_LABELS)
            curb_gt = lbl == CURB_LABEL
            data.append({
                'pts': pts, 'gt_mask': gt, 'valid_mask': valid,
                'curb_gt': curb_gt, 'seq': seq, 'scan_id': sid,
            })
        print(f"OK ({time.time()-t0:.1f}s) | {len(scan_ids)} frames")
    return data


# ============================================================================
# FASE 1: Wall Rejection Grid Search
# ============================================================================

def _eval_wr_combo(args):
    slope, dz, radius = args
    config = PipelineConfig(
        th_dist=0.125,
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=slope,
        wall_height_diff_threshold=dz,
        wall_kdtree_radius=radius,
        enable_curb_detection=False,
        enable_delta_r=False,
        verbose=False,
    )
    pipe = LidarPipelineSuite(config)
    tp, fp, fn = 0, 0, 0
    for frame in _GLOBAL_TRAIN:
        result = pipe.stage2_complete(frame['pts'])
        t, f, n = compute_metrics(frame['gt_mask'], result['obs_mask'], frame['valid_mask'])
        tp += t; fp += f; fn += n
    f1, iou, p, r = metrics_from_accum(tp, fp, fn)
    return {'slope': slope, 'dz': dz, 'radius': radius,
            'f1': f1, 'iou': iou, 'p': p, 'r': r, 'tp': tp, 'fp': fp, 'fn': fn}


def phase1_wr_grid_search(workers, top_n=30):
    global _GLOBAL_TRAIN
    grid = {
        'slope': [0.85, 0.90, 0.95, 1.0],
        'dz': [0.10, 0.12, 0.15, 0.18, 0.20],
        'radius': [0.15, 0.20, 0.25, 0.30, 0.35],
    }
    combos = list(product(grid['slope'], grid['dz'], grid['radius']))
    n_combos = len(combos)

    print(f"\n{'='*100}")
    print(f"FASE 1: WALL REJECTION GRID SEARCH ({n_combos} combos)")
    print(f"{'='*100}")

    t0 = time.time()
    with Pool(min(workers, n_combos)) as pool:
        results = pool.map(_eval_wr_combo, combos)
    elapsed = time.time() - t0
    print(f"  {n_combos} combos en {elapsed:.1f}s")

    # Ordenar por F1
    results.sort(key=lambda x: x['f1'], reverse=True)

    # Imprimir top 20
    print(f"\n  TOP 20 — TRAIN")
    print(f"  {'#':>3}  {'slope':>5}  {'dz':>5}  {'r':>5} | {'F1':>7}  {'IoU':>7}  {'P':>7}  {'R':>7}")
    print(f"  {'-'*65}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3}  {r['slope']:>5.2f}  {r['dz']:>5.2f}  {r['radius']:>5.2f} | "
              f"{100*r['f1']:>6.2f}%  {100*r['iou']:>6.2f}%  {100*r['p']:>6.2f}%  {100*r['r']:>6.2f}%")

    best = results[0]
    print(f"\n  MEJOR: slope={best['slope']}, dz={best['dz']}, radius={best['radius']}")
    print(f"  F1={100*best['f1']:.2f}%")

    return best


def eval_on_val(config):
    """Evalúa una config en val."""
    pipe = LidarPipelineSuite(config)
    tp, fp, fn = 0, 0, 0
    curb_det, curb_tot = 0, 0
    for frame in _GLOBAL_VAL:
        result = pipe.stage2_complete(frame['pts'])
        t, f, n = compute_metrics(frame['gt_mask'], result['obs_mask'], frame['valid_mask'])
        tp += t; fp += f; fn += n
        d, t2 = curb_recall_from_data(frame['curb_gt'], result['obs_mask'], frame['valid_mask'])
        curb_det += d; curb_tot += t2
    f1, iou, p, r = metrics_from_accum(tp, fp, fn)
    cr = curb_det / curb_tot if curb_tot > 0 else 0.0
    return f1, iou, p, r, cr


# ============================================================================
# FASE 2: Feature 1 (Curb Detection) Grid Search
# ============================================================================

def _eval_curb_combo(args):
    curb_min, curb_max, consecutive, wr_params = args
    config = PipelineConfig(
        th_dist=0.125,
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=wr_params['slope'],
        wall_height_diff_threshold=wr_params['dz'],
        wall_kdtree_radius=wr_params['radius'],
        enable_curb_detection=True,
        curb_height_min=curb_min,
        curb_height_max=curb_max,
        curb_min_consecutive=consecutive,
        enable_delta_r=False,
        verbose=False,
    )
    pipe = LidarPipelineSuite(config)
    tp, fp, fn = 0, 0, 0
    curb_det, curb_tot = 0, 0
    for frame in _GLOBAL_TRAIN:
        result = pipe.stage2_complete(frame['pts'])
        t, f, n = compute_metrics(frame['gt_mask'], result['obs_mask'], frame['valid_mask'])
        tp += t; fp += f; fn += n
        d, t2 = curb_recall_from_data(frame['curb_gt'], result['obs_mask'], frame['valid_mask'])
        curb_det += d; curb_tot += t2
    f1, iou, p, r = metrics_from_accum(tp, fp, fn)
    cr = curb_det / curb_tot if curb_tot > 0 else 0.0
    return {'curb_min': curb_min, 'curb_max': curb_max, 'consecutive': consecutive,
            'f1': f1, 'iou': iou, 'p': p, 'r': r, 'curb_recall': cr}


def phase2_curb_grid_search(wr_params, workers, top_n=30):
    grid = {
        'curb_min': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
        'curb_max': [0.18, 0.20, 0.25, 0.30, 0.35],
        'consecutive': [2, 3, 4, 5],
    }
    combos = [(cm, cx, con, wr_params)
              for cm, cx, con in product(grid['curb_min'], grid['curb_max'], grid['consecutive'])]
    n_combos = len(combos)

    print(f"\n{'='*100}")
    print(f"FASE 2: FEATURE 1 (CURB) GRID SEARCH ({n_combos} combos)")
    print(f"  WR fijo: slope={wr_params['slope']}, dz={wr_params['dz']}, r={wr_params['radius']}")
    print(f"{'='*100}")

    t0 = time.time()
    with Pool(min(workers, n_combos)) as pool:
        results = pool.map(_eval_curb_combo, combos)
    elapsed = time.time() - t0
    print(f"  {n_combos} combos en {elapsed:.1f}s")

    results.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n  TOP 20 — TRAIN")
    print(f"  {'#':>3}  {'cmin':>5}  {'cmax':>5}  {'con':>3} | {'F1':>7}  {'IoU':>7}  {'P':>7}  {'R':>7}  {'CurbR':>7}")
    print(f"  {'-'*75}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3}  {r['curb_min']:>5.2f}  {r['curb_max']:>5.2f}  {r['consecutive']:>3} | "
              f"{100*r['f1']:>6.2f}%  {100*r['iou']:>6.2f}%  {100*r['p']:>6.2f}%  {100*r['r']:>6.2f}%  "
              f"{100*r['curb_recall']:>6.1f}%")

    best = results[0]
    print(f"\n  MEJOR: curb_min={best['curb_min']}, curb_max={best['curb_max']}, "
          f"consecutive={best['consecutive']}")
    print(f"  F1={100*best['f1']:.2f}%  Curb Recall={100*best['curb_recall']:.1f}%")

    return best


# ============================================================================
# FASE 3: Delta-r Grid Search
# ============================================================================

def _eval_delta_r_combo(args):
    thr_obs, thr_void, min_nz, wr_params, curb_params = args
    config = PipelineConfig(
        th_dist=0.125,
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=wr_params['slope'],
        wall_height_diff_threshold=wr_params['dz'],
        wall_kdtree_radius=wr_params['radius'],
        enable_curb_detection=True,
        curb_height_min=curb_params['curb_min'],
        curb_height_max=curb_params['curb_max'],
        curb_min_consecutive=curb_params['consecutive'],
        enable_delta_r=True,
        threshold_obs=thr_obs,
        threshold_void=thr_void,
        delta_r_min_nz=min_nz,
        verbose=False,
    )
    pipe = LidarPipelineSuite(config)
    tp, fp, fn = 0, 0, 0
    curb_det, curb_tot = 0, 0
    for frame in _GLOBAL_TRAIN:
        result = pipe.stage2_complete(frame['pts'])
        t, f, n = compute_metrics(frame['gt_mask'], result['obs_mask'], frame['valid_mask'])
        tp += t; fp += f; fn += n
        d, t2 = curb_recall_from_data(frame['curb_gt'], result['obs_mask'], frame['valid_mask'])
        curb_det += d; curb_tot += t2
    f1, iou, p, r = metrics_from_accum(tp, fp, fn)
    cr = curb_det / curb_tot if curb_tot > 0 else 0.0
    return {'thr_obs': thr_obs, 'thr_void': thr_void, 'min_nz': min_nz,
            'f1': f1, 'iou': iou, 'p': p, 'r': r, 'curb_recall': cr}


def phase3_delta_r_grid_search(wr_params, curb_params, workers, top_n=30):
    grid = {
        'thr_obs': [-0.3, -0.5, -0.6, -0.8],
        'thr_void': [0.8, 1.0, 1.2, 1.5],
        'min_nz': [0.85, 0.90, 0.95],
    }
    combos = [(to, tv, nz, wr_params, curb_params)
              for to, tv, nz in product(grid['thr_obs'], grid['thr_void'], grid['min_nz'])]
    n_combos = len(combos)

    print(f"\n{'='*100}")
    print(f"FASE 3: DELTA-R GRID SEARCH ({n_combos} combos)")
    print(f"  WR fijo: slope={wr_params['slope']}, dz={wr_params['dz']}, r={wr_params['radius']}")
    print(f"  Curb fijo: min={curb_params['curb_min']}, max={curb_params['curb_max']}, "
          f"con={curb_params['consecutive']}")
    print(f"{'='*100}")

    t0 = time.time()
    with Pool(min(workers, n_combos)) as pool:
        results = pool.map(_eval_delta_r_combo, combos)
    elapsed = time.time() - t0
    print(f"  {n_combos} combos en {elapsed:.1f}s")

    results.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n  TOP 20 — TRAIN")
    print(f"  {'#':>3}  {'thr_obs':>7}  {'thr_void':>8}  {'min_nz':>6} | "
          f"{'F1':>7}  {'IoU':>7}  {'P':>7}  {'R':>7}  {'CurbR':>7}")
    print(f"  {'-'*80}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3}  {r['thr_obs']:>7.1f}  {r['thr_void']:>8.1f}  {r['min_nz']:>6.2f} | "
              f"{100*r['f1']:>6.2f}%  {100*r['iou']:>6.2f}%  {100*r['p']:>6.2f}%  {100*r['r']:>6.2f}%  "
              f"{100*r['curb_recall']:>6.1f}%")

    best = results[0]
    print(f"\n  MEJOR: thr_obs={best['thr_obs']}, thr_void={best['thr_void']}, min_nz={best['min_nz']}")
    print(f"  F1={100*best['f1']:.2f}%  Curb Recall={100*best['curb_recall']:.1f}%")

    return best


# ============================================================================
# MAIN
# ============================================================================

def main():
    global _GLOBAL_TRAIN, _GLOBAL_VAL

    parser = argparse.ArgumentParser(description='Grid Search 3D-Curb (3 fases)')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--velodyne_dir', type=str, default=None)
    parser.add_argument('--phase', type=int, default=0,
                        help='Fase a ejecutar (0=todas, 1=WR, 2=Curb, 3=Delta-r)')
    # Params fijos para fases 2 y 3
    parser.add_argument('--wall_slope', type=float, default=None)
    parser.add_argument('--wall_dz', type=float, default=None)
    parser.add_argument('--wall_radius', type=float, default=None)
    parser.add_argument('--curb_min', type=float, default=None)
    parser.add_argument('--curb_max', type=float, default=None)
    parser.add_argument('--curb_consecutive', type=int, default=None)
    args = parser.parse_args()

    print("=" * 100)
    print("GRID SEARCH 3D-CURB — Optimización secuencial")
    print(f"Workers: {args.workers} | Stride: {args.stride}")
    print("=" * 100)

    # Cargar datos
    print(f"\n--- TRAIN ---")
    _GLOBAL_TRAIN = load_all_data(TRAIN_SEQS, args.stride, args.velodyne_dir, args.base_dir)
    print(f"  Total TRAIN: {len(_GLOBAL_TRAIN)} frames")

    print(f"\n--- VAL ---")
    _GLOBAL_VAL = load_all_data(VAL_SEQS, args.stride, args.velodyne_dir, args.base_dir)
    print(f"  Total VAL: {len(_GLOBAL_VAL)} frames")

    # ============================
    # FASE 1: WR
    # ============================
    if args.phase in [0, 1]:
        best_wr = phase1_wr_grid_search(args.workers)
        wr_params = {'slope': best_wr['slope'], 'dz': best_wr['dz'], 'radius': best_wr['radius']}

        # Evaluar en val
        print(f"\n  Evaluando mejor WR en VAL...")
        config_wr = PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=wr_params['slope'],
            wall_height_diff_threshold=wr_params['dz'],
            wall_kdtree_radius=wr_params['radius'],
            enable_curb_detection=False, enable_delta_r=False, verbose=False,
        )
        f1, iou, p, r, cr = eval_on_val(config_wr)
        print(f"  VAL: F1={100*f1:.2f}%  IoU={100*iou:.2f}%  P={100*p:.2f}%  R={100*r:.2f}%")
    else:
        wr_params = {
            'slope': args.wall_slope or 0.95,
            'dz': args.wall_dz or 0.15,
            'radius': args.wall_radius or 0.15,
        }

    # ============================
    # FASE 2: Feature 1
    # ============================
    if args.phase in [0, 2]:
        best_curb = phase2_curb_grid_search(wr_params, args.workers)
        curb_params = {'curb_min': best_curb['curb_min'], 'curb_max': best_curb['curb_max'],
                       'consecutive': best_curb['consecutive']}

        # Evaluar en val
        print(f"\n  Evaluando mejor WR+Curb en VAL...")
        config_curb = PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=wr_params['slope'],
            wall_height_diff_threshold=wr_params['dz'],
            wall_kdtree_radius=wr_params['radius'],
            enable_curb_detection=True,
            curb_height_min=curb_params['curb_min'],
            curb_height_max=curb_params['curb_max'],
            curb_min_consecutive=curb_params['consecutive'],
            enable_delta_r=False, verbose=False,
        )
        f1, iou, p, r, cr = eval_on_val(config_curb)
        print(f"  VAL: F1={100*f1:.2f}%  IoU={100*iou:.2f}%  Curb Recall={100*cr:.1f}%")
    else:
        curb_params = {
            'curb_min': args.curb_min or 0.05,
            'curb_max': args.curb_max or 0.30,
            'consecutive': args.curb_consecutive or 3,
        }

    # ============================
    # FASE 3: Delta-r
    # ============================
    if args.phase in [0, 3]:
        best_dr = phase3_delta_r_grid_search(wr_params, curb_params, args.workers)

        # Evaluar en val
        print(f"\n  Evaluando mejor WR+Curb+delta-r en VAL...")
        config_dr = PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=wr_params['slope'],
            wall_height_diff_threshold=wr_params['dz'],
            wall_kdtree_radius=wr_params['radius'],
            enable_curb_detection=True,
            curb_height_min=curb_params['curb_min'],
            curb_height_max=curb_params['curb_max'],
            curb_min_consecutive=curb_params['consecutive'],
            enable_delta_r=True,
            threshold_obs=best_dr['thr_obs'],
            threshold_void=best_dr['thr_void'],
            delta_r_min_nz=best_dr['min_nz'],
            verbose=False,
        )
        f1, iou, p, r, cr = eval_on_val(config_dr)
        print(f"  VAL: F1={100*f1:.2f}%  IoU={100*iou:.2f}%  Curb Recall={100*cr:.1f}%")

    # ============================
    # RESUMEN FINAL
    # ============================
    print(f"\n{'='*100}")
    print("RESUMEN — PARÁMETROS ÓPTIMOS")
    print(f"{'='*100}")
    print(f"  WR:    slope={wr_params['slope']}, dz={wr_params['dz']}, radius={wr_params['radius']}")
    print(f"  Curb:  min={curb_params['curb_min']}, max={curb_params['curb_max']}, "
          f"consecutive={curb_params['consecutive']}")
    if args.phase in [0, 3]:
        print(f"  DR:    thr_obs={best_dr['thr_obs']}, thr_void={best_dr['thr_void']}, "
              f"min_nz={best_dr['min_nz']}")


if __name__ == '__main__':
    main()
