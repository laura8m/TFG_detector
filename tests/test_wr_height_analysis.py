#!/usr/bin/env python3
"""
Analisis de altura de puntos rescatados por Wall Rejection.

Compara PW++ vanilla vs PW++ + WR y analiza la altura sobre el suelo
de los puntos que WR reclasifica de ground a non-ground (obstáculo).

Uso:
    python3 tests/test_wr_height_analysis.py --seq 08 --stride 5
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file, OBSTACLE_LABELS, IGNORE_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='08')
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()

    seq = args.seq
    stride = args.stride

    config_v = PipelineConfig(enable_hybrid_wall_rejection=False, verbose=False)
    config_w = PipelineConfig(verbose=False)
    pipe_v = LidarPipelineSuite(config_v)
    pipe_w = LidarPipelineSuite(config_w)

    # Contar frames
    scan_ids = list(range(0, 99999, stride))
    first_scan = get_scan_file(seq, scan_ids[0])
    if not first_scan.exists():
        print(f"No se encontró la secuencia {seq}")
        return

    # Filtrar scan_ids existentes
    valid_ids = []
    for sid in scan_ids:
        if get_scan_file(seq, sid).exists():
            valid_ids.append(sid)
    scan_ids = valid_ids

    print("=" * 80)
    print(f"ANÁLISIS DE ALTURA — Puntos rescatados por Wall Rejection")
    print(f"Secuencia: {seq} | Frames: {len(scan_ids)} (stride={stride})")
    print("=" * 80)

    all_h = []         # altura sobre suelo
    all_z = []         # Z absoluta
    all_gt_tp = []     # altura de TPs (obstáculo real rescatado)
    all_gt_fp = []     # altura de FPs (ground real rescatado como obstáculo)
    t0 = time.time()

    for i, scan_id in enumerate(scan_ids):
        scan_file = get_scan_file(seq, scan_id)
        label_file = get_label_file(seq, scan_id)

        pts = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]

        r_v = pipe_v.stage2_complete(pts)
        r_w = pipe_w.stage2_complete(pts)

        # Puntos rescatados: non-ground en WR pero ground en vanilla
        rescued = r_w['obs_mask'] & ~r_v['obs_mask']

        if not np.any(rescued):
            continue

        # Z absoluta
        z_rescued = pts[rescued, 2]
        all_z.append(z_rescued)

        # Altura sobre suelo (usando mediana de ground como referencia)
        ground_z = np.median(pts[r_v['ground_mask'], 2]) if np.any(r_v['ground_mask']) else -1.73
        h_rescued = z_rescued - ground_z
        all_h.append(h_rescued)

        # Si hay labels, separar TP de FP
        if label_file.exists():
            labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF
            gt_obs = np.isin(labels, OBSTACLE_LABELS)
            valid = ~np.isin(labels, IGNORE_LABELS)

            rescued_valid = rescued & valid
            tp_mask = rescued_valid & gt_obs
            fp_mask = rescued_valid & ~gt_obs

            if np.any(tp_mask):
                all_gt_tp.append(pts[tp_mask, 2] - ground_z)
            if np.any(fp_mask):
                all_gt_fp.append(pts[fp_mask, 2] - ground_z)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(scan_ids)} frames...")

    elapsed = time.time() - t0
    z = np.concatenate(all_z)
    h = np.concatenate(all_h)

    print(f"\n  Procesado en {elapsed:.1f}s")

    # --- Resultados ---
    print("\n" + "=" * 80)
    print("PUNTOS RESCATADOS POR WALL REJECTION")
    print("=" * 80)
    print(f"  Total: {len(h):,}")
    print(f"  Z absoluta — media: {np.mean(z):.2f}m, mediana: {np.median(z):.2f}m")
    print(f"  Altura sobre suelo — media: {np.mean(h):.3f}m, mediana: {np.median(h):.3f}m")
    print(f"  Z min: {np.min(z):.2f}m, Z max: {np.max(z):.2f}m")

    print(f"\n  Distribucion por altura sobre suelo:")
    bins = [(-0.10, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
            (0.20, 0.30), (0.30, 0.50), (0.50, 1.00), (1.00, 3.00)]
    for lo, hi in bins:
        n = np.sum((h >= lo) & (h < hi))
        print(f"    h [{lo:>5.2f}, {hi:>5.2f})m: {n:>8,} ({n/len(h)*100:>5.1f}%)")

    # --- TP vs FP ---
    if all_gt_tp and all_gt_fp:
        h_tp = np.concatenate(all_gt_tp)
        h_fp = np.concatenate(all_gt_fp)
        print(f"\n" + "=" * 80)
        print("ANÁLISIS TP vs FP (puntos rescatados)")
        print("=" * 80)
        print(f"  TPs (obstáculo real rescatado):  {len(h_tp):>8,} ({len(h_tp)/(len(h_tp)+len(h_fp))*100:.1f}%)")
        print(f"    Altura media: {np.mean(h_tp):.3f}m, mediana: {np.median(h_tp):.3f}m")
        print(f"  FPs (ground real como obstáculo): {len(h_fp):>8,} ({len(h_fp)/(len(h_tp)+len(h_fp))*100:.1f}%)")
        print(f"    Altura media: {np.mean(h_fp):.3f}m, mediana: {np.median(h_fp):.3f}m")

        print(f"\n  Distribucion por altura — TP vs FP:")
        print(f"  {'Rango':<20} {'TP':>8} {'%TP':>6} {'FP':>8} {'%FP':>6}")
        print(f"  {'-'*50}")
        for lo, hi in bins:
            n_tp = np.sum((h_tp >= lo) & (h_tp < hi))
            n_fp = np.sum((h_fp >= lo) & (h_fp < hi))
            pct_tp = n_tp / len(h_tp) * 100 if len(h_tp) > 0 else 0
            pct_fp = n_fp / len(h_fp) * 100 if len(h_fp) > 0 else 0
            print(f"  [{lo:>5.2f}, {hi:>5.2f})m {n_tp:>8,} {pct_tp:>5.1f}% {n_fp:>8,} {pct_fp:>5.1f}%")


if __name__ == '__main__':
    main()
