#!/usr/bin/env python3
"""
Grid search de parámetros de detección de bordillos (Feature 1).

Usa bins completos de SemanticKITTI + labels mapeados con curb=3.
Train: seq 00-07, 09-10 | Val: seq 08
Optimiza F1 global (con curb = obstáculo) y curb recall.

OPTIMIZADO: cachea Stage 1 (Patchwork++ + WR) y solo repite detect_curbs por combo.
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir, get_labels_dir,
                        VELODYNE_ROOT, OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL)

TRAIN_SEQS = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQS = ['08']


def discover_scan_ids(seq, stride=5):
    """Descubre scan IDs disponibles con labels mapeados (curb)."""
    lab_dir = get_labels_dir(seq, use_curb=True)
    vel_dir = get_velodyne_dir(seq)
    if not lab_dir.exists() or not vel_dir.exists():
        return []
    label_files = sorted(lab_dir.glob('*.label'))
    scan_ids = []
    for lf in label_files[::stride]:
        sid = int(lf.stem)
        if (vel_dir / f'{sid:06d}.bin').exists():
            scan_ids.append(sid)
    return scan_ids


def precompute_stage1(seqs, stride):
    """Ejecuta Stage 1 (PW++ + WR) una sola vez y cachea resultados."""
    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_curb_detection=False,
        enable_delta_r=False,
        verbose=False
    )
    pipe = LidarPipelineSuite(config)

    cached = []
    for seq in seqs:
        for sid in discover_scan_ids(seq, stride):
            bin_file = str(get_scan_file(seq, sid))
            lbl_file = str(get_label_file(seq, sid, use_curb=True))

            pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]
            lbl = np.fromfile(lbl_file, dtype=np.uint32) & 0xFFFF

            gt = np.isin(lbl, OBSTACLE_LABELS)
            valid = ~np.isin(lbl, IGNORE_LABELS)
            curb_gt = lbl == CURB_LABEL

            # Stage 1 (una sola vez por frame)
            s1 = pipe.stage1_complete(pts)

            # Máscara base de obstáculos desde Stage 1
            N = len(pts)
            base_obs = np.zeros(N, dtype=bool)
            base_obs[s1['nonground_indices']] = True
            ground_mask = ~base_obs

            cached.append({
                'pts': pts,
                'gt': gt,
                'valid': valid,
                'curb_gt': curb_gt,
                'base_obs': base_obs,
                'ground_mask': ground_mask,
            })

    return cached, pipe


def evaluate_curb_params(pipe, cached, cmin, cmax, consecutive, ring_gap=1):
    """Evalúa un combo de parámetros de curb sobre datos cacheados."""
    pipe.config.enable_curb_detection = True
    pipe.config.curb_height_min = cmin
    pipe.config.curb_height_max = cmax
    pipe.config.curb_min_consecutive = consecutive
    pipe.config.curb_ring_gap = ring_gap

    tp, fp, fn = 0, 0, 0
    curb_tp, curb_total = 0, 0

    for d in cached:
        # Solo ejecutar detect_curbs (Stage 1 ya cacheado)
        curb_mask = pipe.detect_curbs(d['pts'], d['ground_mask'])
        obs_mask = d['base_obs'] | curb_mask

        g = d['gt'] & d['valid']
        p = obs_mask & d['valid']

        tp += int(np.sum(g & p))
        fp += int(np.sum(~g & p))
        fn += int(np.sum(g & ~p))

        n_curb = int(d['curb_gt'].sum())
        if n_curb > 0:
            curb_tp += int(np.sum(d['curb_gt'] & obs_mask))
            curb_total += n_curb

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    curb_recall = curb_tp / curb_total * 100 if curb_total > 0 else 0

    return f1, prec, rec, curb_recall, curb_tp, curb_total


def evaluate_baseline(cached):
    """Evalúa baseline (sin curb detection) sobre datos cacheados."""
    tp, fp, fn = 0, 0, 0
    curb_tp, curb_total = 0, 0

    for d in cached:
        g = d['gt'] & d['valid']
        p = d['base_obs'] & d['valid']

        tp += int(np.sum(g & p))
        fp += int(np.sum(~g & p))
        fn += int(np.sum(g & ~p))

        n_curb = int(d['curb_gt'].sum())
        if n_curb > 0:
            curb_tp += int(np.sum(d['curb_gt'] & d['base_obs']))
            curb_total += n_curb

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    curb_recall = curb_tp / curb_total * 100 if curb_total > 0 else 0

    return f1, prec, rec, curb_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()

    print("=" * 100)
    print("GRID SEARCH — DETECCIÓN DE BORDILLOS (con caché de Stage 1)")
    print("=" * 100)

    # Precomputar Stage 1 para train y val
    print("\nPrecomputando Stage 1 (PW++ + WR) para TRAIN...")
    t0 = time.time()
    train_cached, pipe = precompute_stage1(TRAIN_SEQS, args.stride)
    print(f"  {len(train_cached)} frames en {time.time()-t0:.1f}s")

    print("Precomputando Stage 1 para VAL...")
    t0 = time.time()
    val_cached, _ = precompute_stage1(VAL_SEQS, args.stride)
    print(f"  {len(val_cached)} frames en {time.time()-t0:.1f}s")

    # Baselines
    print("\n--- BASELINES (PW++ + WR, sin curb detection) ---")
    f1_t, p_t, r_t, cr_t = evaluate_baseline(train_cached)
    f1_v, p_v, r_v, cr_v = evaluate_baseline(val_cached)
    print(f"  Train: F1={f1_t:.2f}% P={p_t:.2f}% R={r_t:.2f}% CurbR={cr_t:.1f}%")
    print(f"  Val:   F1={f1_v:.2f}% P={p_v:.2f}% R={r_v:.2f}% CurbR={cr_v:.1f}%")

    # Grid search con ring_gap
    curb_min_vals = [0.05, 0.08, 0.10, 0.12, 0.15]
    curb_max_vals = [0.20, 0.25, 0.30]
    consecutive_vals = [2, 3]
    ring_gap_vals = [1, 2, 3, 4]

    combos = list(product(curb_min_vals, curb_max_vals, consecutive_vals, ring_gap_vals))
    combos = [(cmin, cmax, cons, gap) for cmin, cmax, cons, gap in combos if cmin < cmax]

    print(f"\n--- GRID SEARCH: {len(combos)} combos en TRAIN (solo detect_curbs) ---")
    t0 = time.time()

    train_results = []
    for i, (cmin, cmax, cons, gap) in enumerate(combos):
        f1, prec, rec, curb_r, _, _ = evaluate_curb_params(
            pipe, train_cached, cmin, cmax, cons, ring_gap=gap)
        train_results.append((cmin, cmax, cons, gap, f1, prec, rec, curb_r))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  {i+1}/{len(combos)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  {len(combos)} combos en {elapsed:.0f}s ({elapsed/len(combos):.1f}s/combo)")

    # Ordenar por F1
    train_results.sort(key=lambda x: x[4], reverse=True)

    print(f"\n--- TOP 15 TRAIN (por F1) ---")
    print(f"  {'#':>3} {'cmin':>6} {'cmax':>6} {'cons':>5} {'gap':>4} | {'F1':>8} {'P':>8} {'R':>8} {'CurbR':>8}")
    print(f"  {'-'*68}")
    for i, (cmin, cmax, cons, gap, f1, prec, rec, curb_r) in enumerate(train_results[:15]):
        print(f"  {i+1:>3} {cmin:>6.2f} {cmax:>6.2f} {cons:>5} {gap:>4} | {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}% {curb_r:>7.1f}%")

    # Ordenar por CurbRecall
    by_curb = sorted(train_results, key=lambda x: x[7], reverse=True)
    print(f"\n--- TOP 15 TRAIN (por CurbRecall) ---")
    print(f"  {'#':>3} {'cmin':>6} {'cmax':>6} {'cons':>5} {'gap':>4} | {'F1':>8} {'P':>8} {'R':>8} {'CurbR':>8}")
    print(f"  {'-'*68}")
    for i, (cmin, cmax, cons, gap, f1, prec, rec, curb_r) in enumerate(by_curb[:15]):
        print(f"  {i+1:>3} {cmin:>6.2f} {cmax:>6.2f} {cons:>5} {gap:>4} | {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}% {curb_r:>7.1f}%")

    # Evaluar top 15 en val (por F1 + por CurbRecall, sin duplicados)
    top_combos = []
    seen = set()
    for r in train_results[:15] + by_curb[:15]:
        key = (r[0], r[1], r[2], r[3])
        if key not in seen:
            seen.add(key)
            top_combos.append(r)

    print(f"\n--- EVALUACIÓN EN VAL ({len(top_combos)} combos) ---")
    print(f"  {'#':>3} {'cmin':>6} {'cmax':>6} {'cons':>5} {'gap':>4} | {'F1':>8} {'P':>8} {'R':>8} {'CurbR':>8}")
    print(f"  {'-'*68}")

    best_val_f1 = 0
    best_params = None
    val_results = []

    for i, (cmin, cmax, cons, gap, _, _, _, _) in enumerate(top_combos):
        f1, prec, rec, curb_r, curb_tp, curb_total = evaluate_curb_params(
            pipe, val_cached, cmin, cmax, cons, ring_gap=gap)
        val_results.append((cmin, cmax, cons, gap, f1, prec, rec, curb_r))

    val_results.sort(key=lambda x: x[4], reverse=True)
    for i, (cmin, cmax, cons, gap, f1, prec, rec, curb_r) in enumerate(val_results):
        print(f"  {i+1:>3} {cmin:>6.2f} {cmax:>6.2f} {cons:>5} {gap:>4} | {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}% {curb_r:>7.1f}%")

    best = val_results[0]
    print(f"\n{'='*100}")
    print(f"MEJOR EN VAL: curb_min={best[0]}, curb_max={best[1]}, consecutive={best[2]}, ring_gap={best[3]}")
    print(f"  F1={best[4]:.2f}% P={best[5]:.2f}% R={best[6]:.2f}% CurbRecall={best[7]:.1f}%")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
