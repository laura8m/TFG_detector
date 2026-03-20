#!/usr/bin/env python3
"""
Ablation Study con protocolo SemanticKITTI.

Evalúa la contribución de cada stage del pipeline de forma incremental
en val (seq 08), usando los parámetros óptimos del grid search.

Configuraciones evaluadas:
  1. PW++ vanilla:          non-ground de Patchwork++ = obstáculo
  2. PW++ + Wall Rejection: non-ground después de WR = obstáculo
  3. PW++ + WR + delta-r:   clasificación por anomalía delta-r
  4. PW++ + WR + delta-r + DBSCAN: pipeline completo

Uso:
    # Val (seq 08) con stride=5 (~800 frames)
    python3 tests/test_stage_ablation_semantickitti.py --stride 5

    # Val completo (4071 frames, ~4 min)
    python3 tests/test_stage_ablation_semantickitti.py

    # Con workers para DBSCAN
    python3 tests/test_stage_ablation_semantickitti.py --workers 8
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir,
                        get_labels_dir)

SEMANTICKITTI_VAL = ['08']

OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    50, 51, 52, 70, 71, 80, 81, 99,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)


def discover_scan_ids(seq, stride=1):
    vel_dir = get_velodyne_dir(seq)
    lab_dir = get_labels_dir(seq)
    if not vel_dir.exists() or not lab_dir.exists():
        return []
    vel_ids = {int(f.stem) for f in vel_dir.glob('*.bin')}
    lab_ids = {int(f.stem) for f in lab_dir.glob('*.label')}
    all_ids = sorted(vel_ids & lab_ids)
    return all_ids[::stride]


def load_scan(scan_id, seq):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF
    gt_mask = np.isin(semantic_labels, OBSTACLE_LABELS)
    return points, gt_mask


def compute_metrics(gt_mask, pred_mask):
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1, 'iou': iou}


def replay_dbscan(points, obs_mask, eps, min_samples, min_pts):
    from sklearn.cluster import DBSCAN as _DBSCAN

    obs_indices = np.where(obs_mask)[0]
    if len(obs_indices) == 0:
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
        valid_mask_obs = np.zeros(len(obs_indices), dtype=bool)

    obs_mask_new = np.zeros(N, dtype=bool)
    obs_mask_new[obs_indices[valid_mask_obs]] = True
    return obs_mask_new


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study — protocolo SemanticKITTI')
    parser.add_argument('--val_seq', type=str, nargs='*', default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    val_seqs = args.val_seq if args.val_seq else SEMANTICKITTI_VAL

    print("=" * 100)
    print("ABLATION STUDY — Protocolo SemanticKITTI")
    print("=" * 100)

    # Descubrir frames
    all_scan_ids = {}
    total_frames = 0
    for seq in val_seqs:
        ids = discover_scan_ids(seq, args.stride)
        if ids:
            all_scan_ids[seq] = ids
            total_frames += len(ids)
    print(f"\nVal: {list(all_scan_ids.keys())} | {total_frames} frames | stride={args.stride}")

    # Configuraciones a evaluar
    configs = {
        'PW++ vanilla': PipelineConfig(
            enable_hybrid_wall_rejection=False,
            verbose=False,
        ),
        'PW++ + Wall Rejection': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=0.9,
            wall_height_diff_threshold=0.2,
            wall_kdtree_radius=0.3,
            verbose=False,
        ),
        'PW++ + WR + delta-r': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=0.9,
            wall_height_diff_threshold=0.2,
            wall_kdtree_radius=0.3,
            threshold_obs=-0.4,
            threshold_void=1.2,
            verbose=False,
        ),
        'PW++ + WR + DBSCAN': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=0.9,
            wall_height_diff_threshold=0.2,
            wall_kdtree_radius=0.3,
            verbose=False,
        ),
        'PW++ + WR + delta-r + DBSCAN': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            wall_rejection_slope=0.9,
            wall_height_diff_threshold=0.2,
            wall_kdtree_radius=0.3,
            threshold_obs=-0.4,
            threshold_void=1.2,
            cluster_eps=0.8,
            cluster_min_samples=12,
            cluster_min_pts=30,
            enable_cluster_filtering=True,
            verbose=False,
        ),
    }

    # DBSCAN params para configs sin DBSCAN integrado
    dbscan_params = {
        'eps': 0.8,
        'min_samples': 12,
        'min_pts': 30,
    }

    # Cargar todos los frames
    print(f"\nCargando frames...")
    frames = []
    for seq, scan_ids in all_scan_ids.items():
        t0 = time.time()
        print(f"  Seq {seq}: {len(scan_ids)} frames...", end=" ", flush=True)
        for scan_id in scan_ids:
            pts, gt_mask = load_scan(scan_id, seq)
            frames.append((pts, gt_mask))
        print(f"OK ({time.time()-t0:.1f}s)")

    print(f"\n  Total: {len(frames)} frames")

    # Evaluar cada configuración
    results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*100}")
        print(f"Evaluando: {config_name}")
        print(f"{'='*100}")

        pipeline = LidarPipelineSuite(config)
        total_tp, total_fp, total_fn = 0, 0, 0
        total_time_ms = 0

        for i, (pts, gt_mask) in enumerate(frames):
            t0 = time.time()

            if config_name == 'PW++ vanilla':
                # Solo Patchwork++: non-ground = obstáculo
                pipeline.patchwork.estimateGround(pts)
                ground_idx = set(pipeline.patchwork.getGroundIndices())
                pred_mask = np.array([i not in ground_idx for i in range(len(pts))], dtype=bool)

            elif config_name == 'PW++ + Wall Rejection':
                # PW++ + WR: nonground_indices (ya incluye paredes rechazadas)
                s1 = pipeline.stage1_complete(pts)
                pred_mask = np.zeros(len(pts), dtype=bool)
                pred_mask[s1['nonground_indices']] = True

            elif config_name == 'PW++ + WR + DBSCAN':
                # PW++ + WR + DBSCAN (sin delta-r)
                s1 = pipeline.stage1_complete(pts)
                obs_mask = np.zeros(len(pts), dtype=bool)
                obs_mask[s1['nonground_indices']] = True
                pred_mask = replay_dbscan(
                    pts, obs_mask,
                    dbscan_params['eps'],
                    dbscan_params['min_samples'],
                    dbscan_params['min_pts'],
                )

            elif config_name == 'PW++ + WR + delta-r':
                # Pipeline Stage 1+2 (sin DBSCAN)
                result = pipeline.stage2_complete(pts)
                pred_mask = result['obs_mask']

            elif config_name == 'PW++ + WR + delta-r + DBSCAN':
                # Pipeline completo
                result = pipeline.stage2_complete(pts)
                pred_mask = replay_dbscan(
                    pts, result['obs_mask'],
                    dbscan_params['eps'],
                    dbscan_params['min_samples'],
                    dbscan_params['min_pts'],
                )

            t_ms = (time.time() - t0) * 1000.0
            total_time_ms += t_ms

            tp, fp, fn = int(np.sum(gt_mask & pred_mask)), \
                         int(np.sum((~gt_mask) & pred_mask)), \
                         int(np.sum(gt_mask & (~pred_mask)))
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if (i + 1) % max(1, len(frames) // 10) == 0:
                print(f"\r  [{i+1}/{len(frames)}] {100*(i+1)/len(frames):.0f}%", end="", flush=True)

        print(f"\r  {len(frames)} frames en {total_time_ms/1000:.1f}s "
              f"({total_time_ms/len(frames):.1f} ms/frame)")

        m = compute_metrics(
            np.ones(1, dtype=bool),  # dummy
            np.ones(1, dtype=bool),  # dummy
        )
        # Calcular métricas reales
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        results[config_name] = {
            'f1': f1, 'iou': iou, 'precision': p, 'recall': r,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'ms_per_frame': total_time_ms / len(frames),
        }

        print(f"  F1={100*f1:.2f}%  IoU={100*iou:.2f}%  P={100*p:.2f}%  R={100*r:.2f}%  "
              f"({total_time_ms/len(frames):.1f} ms/frame)")

    # Tabla resumen
    print(f"\n{'='*100}")
    print("ABLATION STUDY — RESUMEN")
    print(f"Val: {list(all_scan_ids.keys())} | {total_frames} frames | stride={args.stride}")
    print(f"{'='*100}")

    print(f"\n{'Configuracion':<40} | {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8} {'ms/frame':>10}")
    print("-" * 95)

    prev_f1 = None
    for name, r in results.items():
        delta = ""
        if prev_f1 is not None:
            df1 = r['f1'] - prev_f1
            delta = f" ({100*df1:+.2f}%)"
        print(f"{name:<40} | {100*r['f1']:>7.2f}% {100*r['iou']:>7.2f}% "
              f"{100*r['precision']:>7.2f}% {100*r['recall']:>7.2f}% "
              f"{r['ms_per_frame']:>9.1f}{delta}")
        prev_f1 = r['f1']

    print(f"\nMejora total: +{100*(results['PW++ + WR + delta-r + DBSCAN']['f1'] - results['PW++ vanilla']['f1']):.2f}% F1")


if __name__ == '__main__':
    main()
