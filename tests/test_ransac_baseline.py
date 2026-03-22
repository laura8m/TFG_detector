#!/usr/bin/env python3
"""
RANSAC Baseline: plano global único para ground segmentation.
Compara RANSAC simple vs PW++ vanilla vs PW++ + WR.
Evaluación en SemanticKITTI val (seq 08).

Uso:
    python3 tests/test_ransac_baseline.py --seq 08 --stride 5
    python3 tests/test_ransac_baseline.py --seq 00 --stride 10  # local rápido
"""
import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_paths import get_sequence_paths
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# === Labels SemanticKITTI ===
OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    50, 51,
    70, 71,
    80, 81,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)

IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)


def ransac_ground_segmentation(points, n_iterations=100, distance_threshold=0.3,
                                z_max_ground=-1.0):
    """
    RANSAC simple: ajusta un plano global al suelo.
    Solo considera puntos con Z < z_max_ground como candidatos.
    """
    # Filtrar candidatos a ground (puntos bajos)
    candidates = points[points[:, 2] < z_max_ground]

    if len(candidates) < 3:
        return np.zeros(len(points), dtype=bool)

    best_inliers = 0
    best_plane = None

    for _ in range(n_iterations):
        # Seleccionar 3 puntos aleatorios
        idx = np.random.choice(len(candidates), 3, replace=False)
        p1, p2, p3 = candidates[idx[0], :3], candidates[idx[1], :3], candidates[idx[2], :3]

        # Calcular normal del plano
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Solo aceptar planos casi horizontales
        if abs(normal[2]) < 0.8:
            continue

        d = -np.dot(normal, p1)

        # Contar inliers en candidatos
        distances = np.abs(np.dot(candidates[:, :3], normal) + d)
        n_inliers = np.sum(distances < distance_threshold)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = (normal, d)

    if best_plane is None:
        return np.zeros(len(points), dtype=bool)

    normal, d = best_plane
    distances = np.abs(np.dot(points[:, :3], normal) + d)
    ground_mask = distances < distance_threshold

    return ground_mask


def load_scan(scan_id, seq, paths):
    """Carga un scan y sus labels."""
    velodyne_dir, label_dir = paths
    bin_file = os.path.join(velodyne_dir, f"{scan_id:06d}.bin")
    label_file = os.path.join(label_dir, f"{scan_id:06d}.label")

    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF

    gt_obs = np.isin(labels, OBSTACLE_LABELS)
    valid_mask = ~np.isin(labels, IGNORE_LABELS)

    return points, gt_obs, valid_mask


def compute_metrics(gt_obs, pred_obs, valid_mask):
    """Calcula TP, FP, FN con valid_mask."""
    gt = gt_obs & valid_mask
    pred = pred_obs & valid_mask
    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default='00')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--ransac_iters', type=int, default=200)
    parser.add_argument('--ransac_dist', type=float, default=0.3)
    args = parser.parse_args()

    seq = args.seq
    paths = get_sequence_paths(seq)

    # Contar frames disponibles
    velodyne_dir = paths[0]
    n_total = len([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    scan_ids = list(range(0, n_total, args.stride))
    n_frames = len(scan_ids)

    print("=" * 80)
    print(f"COMPARACIÓN: RANSAC baseline vs PW++ vanilla vs PW++ + WR")
    print(f"Secuencia: {seq} | Frames: {n_frames} (stride={args.stride})")
    print("=" * 80)

    # === RANSAC ===
    print(f"\nEvaluando: RANSAC global (iters={args.ransac_iters}, dist={args.ransac_dist}m)...")
    tp_r, fp_r, fn_r = 0, 0, 0
    t0 = time.time()
    for scan_id in scan_ids:
        points, gt_obs, valid_mask = load_scan(scan_id, seq, paths)
        ground_mask = ransac_ground_segmentation(
            points, n_iterations=args.ransac_iters,
            distance_threshold=args.ransac_dist
        )
        pred_obs = ~ground_mask  # non-ground = obstáculo
        tp, fp, fn = compute_metrics(gt_obs, pred_obs, valid_mask)
        tp_r += tp
        fp_r += fp
        fn_r += fn
    t_ransac = (time.time() - t0) / n_frames * 1000

    # === PW++ vanilla ===
    print("Evaluando: PW++ vanilla...")
    config_pw = PipelineConfig(enable_wall_rejection=False, enable_delta_r=False)
    pipe_pw = LidarPipelineSuite(config_pw)
    tp_pw, fp_pw, fn_pw = 0, 0, 0
    t0 = time.time()
    for scan_id in scan_ids:
        points, gt_obs, valid_mask = load_scan(scan_id, seq, paths)
        result = pipe_pw.stage2_complete(points)
        pred_obs = result['obs_mask']
        tp, fp, fn = compute_metrics(gt_obs, pred_obs, valid_mask)
        tp_pw += tp
        fp_pw += fp
        fn_pw += fn
    t_pw = (time.time() - t0) / n_frames * 1000

    # === PW++ + WR ===
    print("Evaluando: PW++ + Wall Rejection...")
    config_wr = PipelineConfig(enable_delta_r=False)
    pipe_wr = LidarPipelineSuite(config_wr)
    tp_wr, fp_wr, fn_wr = 0, 0, 0
    t0 = time.time()
    for scan_id in scan_ids:
        points, gt_obs, valid_mask = load_scan(scan_id, seq, paths)
        result = pipe_wr.stage2_complete(points)
        pred_obs = result['obs_mask']
        tp, fp, fn = compute_metrics(gt_obs, pred_obs, valid_mask)
        tp_wr += tp
        fp_wr += fp
        fn_wr += fn
    t_wr = (time.time() - t0) / n_frames * 1000

    # === RESULTADOS ===
    def calc(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        return f1, iou, p, r

    f1_r, iou_r, p_r, r_r = calc(tp_r, fp_r, fn_r)
    f1_pw, iou_pw, p_pw, r_pw = calc(tp_pw, fp_pw, fn_pw)
    f1_wr, iou_wr, p_wr, r_wr = calc(tp_wr, fp_wr, fn_wr)

    print(f"\n{'='*80}")
    print(f"RESULTADOS — Seq {seq} ({n_frames} frames, stride={args.stride})")
    print(f"{'='*80}")
    print(f"  {'Config':<30} {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8} {'ms':>8}")
    print(f"  {'-'*72}")
    print(f"  {'RANSAC global':<30} {100*f1_r:>7.2f}% {100*iou_r:>7.2f}% {100*p_r:>7.2f}% {100*r_r:>7.2f}% {t_ransac:>7.1f}")
    print(f"  {'PW++ vanilla':<30} {100*f1_pw:>7.2f}% {100*iou_pw:>7.2f}% {100*p_pw:>7.2f}% {100*r_pw:>7.2f}% {t_pw:>7.1f}")
    print(f"  {'PW++ + Wall Rejection':<30} {100*f1_wr:>7.2f}% {100*iou_wr:>7.2f}% {100*p_wr:>7.2f}% {100*r_wr:>7.2f}% {t_wr:>7.1f}")

    print(f"\n  Mejora PW++ vs RANSAC:   +{100*(f1_pw-f1_r):.2f}% F1")
    print(f"  Mejora WR vs PW++:       +{100*(f1_wr-f1_pw):.2f}% F1")
    print(f"  Mejora WR vs RANSAC:     +{100*(f1_wr-f1_r):.2f}% F1")


if __name__ == '__main__':
    main()
