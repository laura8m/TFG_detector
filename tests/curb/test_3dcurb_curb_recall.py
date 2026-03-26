#!/usr/bin/env python3
"""
Análisis detallado de detección de bordillos por stage.

Muestra cuántos puntos de bordillo (label 3) rescata cada stage del pipeline,
con desglose por distancia al sensor y altura.

Uso:
    python3 tests/curb/test_3dcurb_curb_recall.py --stride 5
    python3 tests/curb/test_3dcurb_curb_recall.py --stride 5 --base_dir /path/to/3d_curb_labels
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent))
from data_paths_3dcurb import (
    discover_scan_ids, OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL,
    VAL_SEQS, get_velodyne_dir, get_labels_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description='Curb Recall por stage — 3D-Curb')
    parser.add_argument('--seq', type=str, default='08')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--velodyne_dir', type=str, default=None)
    # Params
    parser.add_argument('--wall_slope', type=float, default=0.95)
    parser.add_argument('--wall_dz', type=float, default=0.15)
    parser.add_argument('--wall_radius', type=float, default=0.15)
    parser.add_argument('--curb_min', type=float, default=0.05)
    parser.add_argument('--curb_max', type=float, default=0.30)
    parser.add_argument('--curb_consecutive', type=int, default=3)
    parser.add_argument('--threshold_obs', type=float, default=-0.8)
    parser.add_argument('--threshold_void', type=float, default=1.5)
    parser.add_argument('--min_nz', type=float, default=0.95)
    args = parser.parse_args()

    seq = args.seq
    scan_ids = discover_scan_ids(seq, args.stride, args.velodyne_dir, args.base_dir)

    print("=" * 100)
    print(f"ANÁLISIS DE DETECCIÓN DE BORDILLOS — 3D-Curb seq {seq}")
    print(f"  Frames: {len(scan_ids)} | stride={args.stride}")
    print("=" * 100)

    # 4 configuraciones incrementales
    configs = {
        'PW++ vanilla': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=False,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False,
        ),
        'PW++ + WR': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=args.wall_slope,
            wall_height_diff_threshold=args.wall_dz,
            wall_kdtree_radius=args.wall_radius,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False,
        ),
        'PW++ + WR + Curb': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=args.wall_slope,
            wall_height_diff_threshold=args.wall_dz,
            wall_kdtree_radius=args.wall_radius,
            enable_curb_detection=True,
            curb_height_min=args.curb_min,
            curb_height_max=args.curb_max,
            curb_min_consecutive=args.curb_consecutive,
            enable_delta_r=False,
            verbose=False,
        ),
        'PW++ + WR + Curb + DR': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=args.wall_slope,
            wall_height_diff_threshold=args.wall_dz,
            wall_kdtree_radius=args.wall_radius,
            enable_curb_detection=True,
            curb_height_min=args.curb_min,
            curb_height_max=args.curb_max,
            curb_min_consecutive=args.curb_consecutive,
            enable_delta_r=True,
            threshold_obs=args.threshold_obs,
            threshold_void=args.threshold_void,
            delta_r_min_nz=args.min_nz,
            verbose=False,
        ),
    }

    # Acumuladores por config
    curb_stats = {name: {'detected': 0, 'total': 0} for name in configs}
    # Acumuladores por distancia
    dist_ranges = [(0, 10), (10, 20), (20, 40), (40, 80)]
    curb_by_dist = {name: {r: {'detected': 0, 'total': 0} for r in dist_ranges} for name in configs}
    # Acumuladores de Z
    curb_z_detected = {name: [] for name in configs}

    vel_dir = get_velodyne_dir(seq, args.velodyne_dir)
    lab_dir = get_labels_dir(seq, args.base_dir)

    pipelines = {name: LidarPipelineSuite(config) for name, config in configs.items()}

    for idx, sid in enumerate(scan_ids):
        pts = np.fromfile(str(vel_dir / f"{sid:06d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl = np.fromfile(str(lab_dir / f"{sid:06d}.label"), dtype=np.uint32) & 0xFFFF

        curb_gt = lbl == CURB_LABEL
        valid = ~np.isin(lbl, IGNORE_LABELS)
        curb_valid = curb_gt & valid

        if not np.any(curb_valid):
            continue

        # Distancia al sensor
        dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

        for name, pipe in pipelines.items():
            if name == 'PW++ vanilla':
                pipe.patchwork.estimateGround(pts)
                ground_idx = set(pipe.patchwork.getGroundIndices())
                obs_mask = np.array([j not in ground_idx for j in range(len(pts))], dtype=bool)
            else:
                result = pipe.stage2_complete(pts)
                obs_mask = result['obs_mask']

            detected = curb_valid & obs_mask
            curb_stats[name]['detected'] += int(np.sum(detected))
            curb_stats[name]['total'] += int(np.sum(curb_valid))

            # Por distancia
            for lo, hi in dist_ranges:
                mask_dist = (dist >= lo) & (dist < hi)
                curb_in_range = curb_valid & mask_dist
                if np.any(curb_in_range):
                    curb_by_dist[name][(lo, hi)]['total'] += int(np.sum(curb_in_range))
                    curb_by_dist[name][(lo, hi)]['detected'] += int(np.sum(curb_in_range & obs_mask))

            # Z de bordillos detectados
            if np.any(detected):
                curb_z_detected[name].append(pts[detected, 2])

        if (idx + 1) % max(1, len(scan_ids) // 10) == 0:
            print(f"\r  [{idx+1}/{len(scan_ids)}] {100*(idx+1)/len(scan_ids):.0f}%", end="", flush=True)

    print(f"\r  {len(scan_ids)} frames procesados")

    # ================================
    # TABLA PRINCIPAL
    # ================================
    print(f"\n{'='*100}")
    print("CURB RECALL POR STAGE")
    print(f"{'='*100}")

    print(f"\n  {'Config':<25} {'Curb Total':>10} {'Detectados':>10} {'Recall':>8} "
          f"{'By WR':>8} {'By Curb':>8} {'By DR':>8}")
    print(f"  {'-'*85}")

    prev_detected = 0
    names = list(configs.keys())
    vanilla_detected = curb_stats[names[0]]['detected']

    for i, name in enumerate(names):
        s = curb_stats[name]
        recall = s['detected'] / s['total'] * 100 if s['total'] > 0 else 0
        by_wr = ""
        by_curb = ""
        by_dr = ""
        if i == 1:
            by_wr = f"+{s['detected'] - vanilla_detected}"
        elif i == 2:
            by_wr = f"+{curb_stats[names[1]]['detected'] - vanilla_detected}"
            by_curb = f"+{s['detected'] - curb_stats[names[1]]['detected']}"
        elif i == 3:
            by_wr = f"+{curb_stats[names[1]]['detected'] - vanilla_detected}"
            by_curb = f"+{curb_stats[names[2]]['detected'] - curb_stats[names[1]]['detected']}"
            by_dr = f"+{s['detected'] - curb_stats[names[2]]['detected']}"

        print(f"  {name:<25} {s['total']:>10} {s['detected']:>10} {recall:>7.1f}% "
              f"{by_wr:>8} {by_curb:>8} {by_dr:>8}")

    # ================================
    # POR DISTANCIA
    # ================================
    print(f"\n{'='*100}")
    print("CURB RECALL POR DISTANCIA")
    print(f"{'='*100}")

    for lo, hi in dist_ranges:
        print(f"\n  Rango {lo}-{hi}m:")
        print(f"    {'Config':<25} {'Total':>8} {'Detectados':>10} {'Recall':>8}")
        print(f"    {'-'*55}")
        for name in names:
            s = curb_by_dist[name][(lo, hi)]
            recall = s['detected'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"    {name:<25} {s['total']:>8} {s['detected']:>10} {recall:>7.1f}%")

    # ================================
    # DISTRIBUCIÓN DE Z
    # ================================
    print(f"\n{'='*100}")
    print("ALTURA Z DE BORDILLOS DETECTADOS")
    print(f"{'='*100}")

    best_name = names[2]  # PW++ + WR + Curb
    if curb_z_detected[best_name]:
        z_all = np.concatenate(curb_z_detected[best_name])
        print(f"\n  Config: {best_name}")
        print(f"  Z media: {np.mean(z_all):.3f}m")
        print(f"  Z mediana: {np.median(z_all):.3f}m")
        print(f"  Z P5={np.percentile(z_all,5):.2f}  P95={np.percentile(z_all,95):.2f}")


if __name__ == '__main__':
    main()
