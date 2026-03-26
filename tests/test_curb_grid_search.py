#!/usr/bin/env python3
"""
Grid search de parámetros de detección de bordillos (Feature 1).

Usa 3D-Curb dataset (label 3 = curb como obstáculo).
Train: seq 00-07, 09-10 | Val: seq 08
Optimiza F1 global (con curb = obstáculo) y curb recall.
"""

import argparse
import numpy as np
import glob
from pathlib import Path
import sys
import time
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL

TRAIN_SEQS = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQS = ['08']


def load_data(seqs, base_dir, stride=5):
    """Carga frames de las secuencias especificadas."""
    frames = []
    for seq in seqs:
        vel_dir = base_dir / seq / 'velodyne'
        lbl_dir = base_dir / seq / 'labels'
        if not vel_dir.exists() or not lbl_dir.exists():
            continue
        label_files = sorted(glob.glob(str(lbl_dir / '*.label')))
        for lf in label_files[::stride]:
            scan_id = int(Path(lf).stem)
            bin_file = vel_dir / f'{scan_id:06d}.bin'
            if not bin_file.exists():
                continue
            frames.append((str(bin_file), str(lf)))
    return frames


def evaluate(pipe, frames):
    """Evalúa F1 global y curb recall."""
    tp, fp, fn = 0, 0, 0
    curb_tp, curb_total = 0, 0

    for bin_file, lbl_file in frames:
        pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl = np.fromfile(lbl_file, dtype=np.uint32) & 0xFFFF

        gt = np.isin(lbl, OBSTACLE_LABELS)
        valid = ~np.isin(lbl, IGNORE_LABELS)
        curb_gt = lbl == CURB_LABEL

        r = pipe.stage2_complete(pts)
        g = gt & valid
        p = r['obs_mask'] & valid

        tp += int(np.sum(g & p))
        fp += int(np.sum(~g & p))
        fn += int(np.sum(g & ~p))

        n_curb = int(curb_gt.sum())
        if n_curb > 0:
            curb_tp += int(np.sum(curb_gt & r['obs_mask']))
            curb_total += n_curb

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    curb_recall = curb_tp / curb_total * 100 if curb_total > 0 else 0

    return f1, prec, rec, curb_recall, curb_tp, curb_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=None, help='Directorio base 3D-Curb')
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        candidates = [
            Path('3d_curb_labels'),
            Path.home() / 'laura' / 'sota_ws' / 'TFG_detector' / '3d_curb_labels',
        ]
        base_dir = None
        for c in candidates:
            if c.exists():
                base_dir = c
                break
        if base_dir is None:
            print("ERROR: No se encontró 3d_curb_labels/")
            sys.exit(1)

    print("=" * 100)
    print("GRID SEARCH — DETECCIÓN DE BORDILLOS (3D-Curb Dataset)")
    print("=" * 100)

    # Cargar datos
    print("\nCargando datos...")
    train_frames = load_data(TRAIN_SEQS, base_dir, args.stride)
    val_frames = load_data(VAL_SEQS, base_dir, args.stride)
    print(f"  Train: {len(train_frames)} frames | Val: {len(val_frames)} frames")

    # Baseline sin curb detection
    print("\n--- BASELINES ---")
    for name, wr in [('PW++ vanilla', False), ('PW++ + WR', True)]:
        config = PipelineConfig(
            enable_hybrid_wall_rejection=wr,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False
        )
        pipe = LidarPipelineSuite(config)
        f1_t, p_t, r_t, cr_t, _, _ = evaluate(pipe, train_frames)
        f1_v, p_v, r_v, cr_v, _, ct_v = evaluate(pipe, val_frames)
        print(f"  {name}:")
        print(f"    Train: F1={f1_t:.2f}% P={p_t:.2f}% R={r_t:.2f}% CurbR={cr_t:.1f}%")
        print(f"    Val:   F1={f1_v:.2f}% P={p_v:.2f}% R={r_v:.2f}% CurbR={cr_v:.1f}%")

    # Grid search
    curb_min_vals = [0.05, 0.08, 0.10, 0.12, 0.15]
    curb_max_vals = [0.20, 0.25, 0.30]
    consecutive_vals = [2, 3, 4]

    combos = list(product(curb_min_vals, curb_max_vals, consecutive_vals))
    # Filtrar combos inválidos (min >= max)
    combos = [(cmin, cmax, cons) for cmin, cmax, cons in combos if cmin < cmax]

    print(f"\n--- GRID SEARCH: {len(combos)} combos en TRAIN ---")
    t0 = time.time()

    train_results = []
    for i, (cmin, cmax, cons) in enumerate(combos):
        config = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=True,
            enable_delta_r=False,
            curb_height_min=cmin,
            curb_height_max=cmax,
            curb_min_consecutive=cons,
            verbose=False
        )
        pipe = LidarPipelineSuite(config)
        f1, prec, rec, curb_r, _, _ = evaluate(pipe, train_frames)
        train_results.append((cmin, cmax, cons, f1, prec, rec, curb_r))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(combos)}...")

    elapsed = time.time() - t0
    print(f"  {len(combos)} combos en {elapsed:.0f}s")

    # Ordenar por F1
    train_results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n--- TOP 10 TRAIN (por F1) ---")
    print(f"  {'#':>3} {'cmin':>6} {'cmax':>6} {'cons':>5} | {'F1':>8} {'P':>8} {'R':>8} {'CurbR':>8}")
    print(f"  {'-'*60}")
    for i, (cmin, cmax, cons, f1, prec, rec, curb_r) in enumerate(train_results[:10]):
        print(f"  {i+1:>3} {cmin:>6.2f} {cmax:>6.2f} {cons:>5} | {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}% {curb_r:>7.1f}%")

    # Evaluar top 10 en val
    print(f"\n--- EVALUACIÓN EN VAL (top 10) ---")
    print(f"  {'#':>3} {'cmin':>6} {'cmax':>6} {'cons':>5} | {'F1':>8} {'P':>8} {'R':>8} {'CurbR':>8}")
    print(f"  {'-'*60}")

    best_val_f1 = 0
    best_params = None

    for i, (cmin, cmax, cons, _, _, _, _) in enumerate(train_results[:10]):
        config = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=True,
            enable_delta_r=False,
            curb_height_min=cmin,
            curb_height_max=cmax,
            curb_min_consecutive=cons,
            verbose=False
        )
        pipe = LidarPipelineSuite(config)
        f1, prec, rec, curb_r, curb_tp, curb_total = evaluate(pipe, val_frames)
        print(f"  {i+1:>3} {cmin:>6.2f} {cmax:>6.2f} {cons:>5} | {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}% {curb_r:>7.1f}%")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_params = (cmin, cmax, cons, f1, prec, rec, curb_r)

    print(f"\n{'='*100}")
    print(f"MEJOR EN VAL: curb_min={best_params[0]}, curb_max={best_params[1]}, consecutive={best_params[2]}")
    print(f"  F1={best_params[3]:.2f}% P={best_params[4]:.2f}% R={best_params[5]:.2f}% CurbRecall={best_params[6]:.1f}%")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
