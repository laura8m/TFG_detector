#!/usr/bin/env python3
"""
Ablation Study con dataset 3D-Curb (SemanticKITTI + curb label 3).

Evalúa la contribución de cada stage del pipeline con bordillos como obstáculo.

Configuraciones evaluadas:
  1. PW++ vanilla:              non-ground = obstáculo
  2. PW++ + Wall Rejection:     non-ground + WR = obstáculo
  3. PW++ + WR + Feature 1:     + detección de bordillos inter-ring
  4. PW++ + WR + F1 + delta-r:  + anomalía delta-r conservadora

Uso:
    python3 tests/curb/test_3dcurb_ablation.py --stride 5
    python3 tests/curb/test_3dcurb_ablation.py --stride 5 --base_dir /path/to/3d_curb_labels
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

# Importar desde el directorio curb/
sys.path.insert(0, str(Path(__file__).parent))
from data_paths_3dcurb import (
    discover_scan_ids, load_scan, OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL,
    VAL_SEQS, get_velodyne_dir, get_labels_dir,
)

# Importar pipeline desde sota_idea/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study — 3D-Curb (bordillos como obstáculo)')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Directorio raíz de labels 3D-Curb')
    parser.add_argument('--velodyne_dir', type=str, default=None,
                        help='Directorio raíz de velodyne .bin')
    # WR params
    parser.add_argument('--wall_slope', type=float, default=0.95)
    parser.add_argument('--wall_dz', type=float, default=0.15)
    parser.add_argument('--wall_radius', type=float, default=0.15)
    # Curb params
    parser.add_argument('--curb_min', type=float, default=0.05)
    parser.add_argument('--curb_max', type=float, default=0.30)
    parser.add_argument('--curb_consecutive', type=int, default=3)
    # Delta-r params
    parser.add_argument('--threshold_obs', type=float, default=-0.8)
    parser.add_argument('--threshold_void', type=float, default=1.5)
    parser.add_argument('--min_nz', type=float, default=0.95)
    args = parser.parse_args()

    print("=" * 100)
    print("ABLATION STUDY — 3D-Curb (bordillos = obstáculo)")
    print("=" * 100)

    # Descubrir frames en val
    all_scan_ids = {}
    total_frames = 0
    for seq in VAL_SEQS:
        ids = discover_scan_ids(seq, args.stride,
                                velodyne_root=args.velodyne_dir,
                                labels_root=args.base_dir)
        if ids:
            all_scan_ids[seq] = ids
            total_frames += len(ids)
    print(f"\nVal: {list(all_scan_ids.keys())} | {total_frames} frames | stride={args.stride}")

    # Cargar frames
    print(f"\nCargando frames...")
    frames = []
    for seq, scan_ids in all_scan_ids.items():
        t0 = time.time()
        print(f"  Seq {seq}: {len(scan_ids)} frames...", end=" ", flush=True)
        for scan_id in scan_ids:
            pts, gt_mask, valid_mask, curb_gt, labels = load_scan(
                seq, scan_id,
                velodyne_root=args.velodyne_dir,
                labels_root=args.base_dir
            )
            frames.append((pts, gt_mask, valid_mask, curb_gt))
        print(f"OK ({time.time()-t0:.1f}s)")
    print(f"  Total: {len(frames)} frames")

    # Configuraciones
    configs = {
        'PW++ vanilla': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=False,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False,
        ),
        'PW++ + Wall Rejection': PipelineConfig(
            th_dist=0.125,
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=args.wall_slope,
            wall_height_diff_threshold=args.wall_dz,
            wall_kdtree_radius=args.wall_radius,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False,
        ),
        'PW++ + WR + Curb (Feature 1)': PipelineConfig(
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
        'PW++ + WR + Curb + delta-r': PipelineConfig(
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

    # Evaluar cada configuración
    results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*100}")
        print(f"Evaluando: {config_name}")
        print(f"{'='*100}")

        pipeline = LidarPipelineSuite(config)
        total_tp, total_fp, total_fn = 0, 0, 0
        total_curb_detected, total_curb_gt = 0, 0
        total_time_ms = 0

        for i, (pts, gt_mask, valid_mask, curb_gt) in enumerate(frames):
            t0 = time.time()

            if config_name == 'PW++ vanilla':
                pipeline.patchwork.estimateGround(pts)
                ground_idx = set(pipeline.patchwork.getGroundIndices())
                pred_mask = np.array([j not in ground_idx for j in range(len(pts))], dtype=bool)
            else:
                result = pipeline.stage2_complete(pts)
                pred_mask = result['obs_mask']

            t_ms = (time.time() - t0) * 1000.0
            total_time_ms += t_ms

            # Métricas con valid_mask
            gt_v = gt_mask & valid_mask
            pred_v = pred_mask & valid_mask
            total_tp += int(np.sum(gt_v & pred_v))
            total_fp += int(np.sum(~gt_v & pred_v))
            total_fn += int(np.sum(gt_v & ~pred_v))

            # Curb recall
            curb_valid = curb_gt & valid_mask
            total_curb_gt += int(np.sum(curb_valid))
            total_curb_detected += int(np.sum(curb_valid & pred_v))

            if (i + 1) % max(1, len(frames) // 10) == 0:
                print(f"\r  [{i+1}/{len(frames)}] {100*(i+1)/len(frames):.0f}%", end="", flush=True)

        print(f"\r  {len(frames)} frames en {total_time_ms/1000:.1f}s "
              f"({total_time_ms/len(frames):.1f} ms/frame)")

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        curb_recall = total_curb_detected / total_curb_gt if total_curb_gt > 0 else 0.0

        results[config_name] = {
            'f1': f1, 'iou': iou, 'precision': p, 'recall': r,
            'curb_recall': curb_recall,
            'curb_detected': total_curb_detected, 'curb_total': total_curb_gt,
            'ms_per_frame': total_time_ms / len(frames),
        }

        print(f"  F1={100*f1:.2f}%  IoU={100*iou:.2f}%  P={100*p:.2f}%  R={100*r:.2f}%  "
              f"Curb={100*curb_recall:.1f}%  ({total_time_ms/len(frames):.1f} ms/frame)")

    # Tabla resumen
    print(f"\n{'='*100}")
    print("ABLATION STUDY — RESUMEN (3D-Curb, bordillos = obstáculo)")
    print(f"Val: {list(all_scan_ids.keys())} | {total_frames} frames | stride={args.stride}")
    print(f"{'='*100}")

    print(f"\n{'Configuracion':<35} | {'F1':>7} {'IoU':>7} {'P':>7} {'R':>7} "
          f"{'Curb R':>7} {'ms/fr':>7}")
    print("-" * 100)

    prev_f1 = None
    config_names = list(results.keys())
    for name in config_names:
        r = results[name]
        delta = ""
        if prev_f1 is not None:
            df1 = r['f1'] - prev_f1
            delta = f" ({100*df1:+.2f}%)"
        print(f"{name:<35} | {100*r['f1']:>6.2f}% {100*r['iou']:>6.2f}% "
              f"{100*r['precision']:>6.2f}% {100*r['recall']:>6.2f}% "
              f"{100*r['curb_recall']:>6.1f}% {r['ms_per_frame']:>6.1f}{delta}")
        prev_f1 = r['f1']

    first = config_names[0]
    last = config_names[-1]
    print(f"\nMejora total: {100*(results[last]['f1'] - results[first]['f1']):+.2f}% F1")
    print(f"Curb recall: {100*results[first]['curb_recall']:.1f}% → {100*results[last]['curb_recall']:.1f}%")


if __name__ == '__main__':
    main()
