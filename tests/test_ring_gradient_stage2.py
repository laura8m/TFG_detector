#!/usr/bin/env python3
"""
Test: Ring Gradient Anomaly Detection (Stage 2 alternativo).

Compara 3 configuraciones:
  1. PW++ + WR (Stage 1 solo, baseline)
  2. PW++ + WR + delta-r clásico (Stage 2 actual)
  3. PW++ + WR + ring gradient (Stage 2 propuesto)

El ring gradient detecta bordes de obstáculos por discontinuidades de rango
entre puntos consecutivos del mismo ring del LiDAR (Velodyne HDL-64E, 64 rings).
A diferencia de delta-r, no depende de los planos RANSAC de Patchwork++.

Modo "solo rescate": el ring gradient solo puede promover ground→nonground,
nunca degrada nonground→ground. Así nunca empeora el baseline.

Uso:
    python3 tests/test_ring_gradient_stage2.py
    python3 tests/test_ring_gradient_stage2.py --seq 00 --n_frames 50
    python3 tests/test_ring_gradient_stage2.py --seq both --n_frames 20
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file, get_velodyne_dir, get_labels_dir

# === SemanticKITTI labels (corregidos) ===
OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    50, 51, 70, 71, 80, 81,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)

IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)

# Velodyne HDL-64E: 64 anillas, FOV vertical -24.33° a +2°
VELODYNE_N_RINGS = 64
VELODYNE_FOV_UP = 2.0       # grados
VELODYNE_FOV_DOWN = -24.33  # grados


def assign_ring_ids(points):
    """
    Asigna cada punto a su ring del Velodyne HDL-64E basándose en el ángulo
    de elevación vertical.

    Args:
        points: (N, 3) XYZ

    Returns:
        ring_ids: (N,) int, 0..63 (-1 si fuera de FOV)
    """
    xy_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    elevation = np.degrees(np.arctan2(points[:, 2], xy_dist))

    fov_total = VELODYNE_FOV_UP - VELODYNE_FOV_DOWN
    ring_ids = np.floor(
        (elevation - VELODYNE_FOV_DOWN) / fov_total * VELODYNE_N_RINGS
    ).astype(np.int32)

    # Clamp a rango válido
    ring_ids = np.clip(ring_ids, 0, VELODYNE_N_RINGS - 1)

    return ring_ids


def compute_ring_gradient(points, ring_ids, gradient_threshold=1.0):
    """
    Detecta bordes de obstáculos por discontinuidades de rango dentro de cada ring.

    Para cada ring, ordena los puntos por azimut y calcula |r[i] - r[i-1]|.
    Un salto > threshold indica borde de obstáculo: el punto más cercano
    es probablemente un obstáculo.

    Args:
        points: (N, 3) XYZ
        ring_ids: (N,) int, ring assignment
        gradient_threshold: umbral de salto de rango (m)

    Returns:
        border_mask: (N,) bool, True para puntos detectados como borde de obstáculo
    """
    N = len(points)
    border_mask = np.zeros(N, dtype=bool)

    # Precomputar rango y azimut
    ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    azimuth = np.arctan2(points[:, 1], points[:, 0])

    for ring in range(VELODYNE_N_RINGS):
        ring_mask = ring_ids == ring
        ring_indices = np.where(ring_mask)[0]

        if len(ring_indices) < 3:
            continue

        # Ordenar por azimut dentro del ring
        az = azimuth[ring_indices]
        sort_order = np.argsort(az)
        sorted_indices = ring_indices[sort_order]
        sorted_ranges = ranges[sorted_indices]

        # Gradiente de rango entre puntos consecutivos
        dr = np.abs(np.diff(sorted_ranges))

        # Puntos donde hay salto grande
        jump_positions = np.where(dr > gradient_threshold)[0]

        for pos in jump_positions:
            idx_before = sorted_indices[pos]
            idx_after = sorted_indices[pos + 1]

            # El punto más cercano es probablemente el obstáculo
            if sorted_ranges[pos] < sorted_ranges[pos + 1]:
                border_mask[idx_before] = True
            else:
                border_mask[idx_after] = True

    return border_mask


def compute_ring_gradient_vectorized(points, ring_ids, gradient_threshold=1.0,
                                     neighbor_window=1):
    """
    Versión vectorizada del ring gradient. Más rápida que el loop por ring.

    Detecta puntos donde el rango cambia bruscamente respecto a vecinos
    del mismo ring. El punto más cercano al sensor en cada discontinuidad
    se marca como borde de obstáculo.

    Args:
        points: (N, 3) XYZ
        ring_ids: (N,) int
        gradient_threshold: umbral de salto (m)
        neighbor_window: cuántos vecinos comparar a cada lado

    Returns:
        border_mask: (N,) bool
    """
    N = len(points)
    border_mask = np.zeros(N, dtype=bool)

    ranges = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    azimuth = np.arctan2(points[:, 1], points[:, 0])

    # Clave compuesta: ring primero, azimut segundo
    # Esto agrupa por ring y ordena por azimut dentro de cada ring
    sort_key = ring_ids.astype(np.float64) * 1000.0 + azimuth
    global_order = np.argsort(sort_key)

    sorted_ring_ids = ring_ids[global_order]
    sorted_ranges = ranges[global_order]

    # Gradiente forward
    dr = np.abs(np.diff(sorted_ranges))
    same_ring = sorted_ring_ids[:-1] == sorted_ring_ids[1:]

    # Posiciones con salto grande dentro del mismo ring
    jumps = np.where((dr > gradient_threshold) & same_ring)[0]

    for pos in jumps:
        idx_before = global_order[pos]
        idx_after = global_order[pos + 1]

        # El punto más cercano es el obstáculo
        if sorted_ranges[pos] < sorted_ranges[pos + 1]:
            border_mask[idx_before] = True
        else:
            border_mask[idx_after] = True

    return border_mask


def load_scan(scan_id, seq):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF
    gt_mask = np.isin(semantic_labels, OBSTACLE_LABELS)
    valid_mask = ~np.isin(semantic_labels, IGNORE_LABELS)
    return points, gt_mask, valid_mask


def compute_metrics(gt_mask, pred_mask, valid_mask=None):
    if valid_mask is not None:
        gt_mask = gt_mask & valid_mask
        pred_mask = pred_mask & valid_mask
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1, 'iou': iou}


def discover_scan_ids(seq, n_frames=None):
    vel_dir = get_velodyne_dir(seq)
    lab_dir = get_labels_dir(seq)
    if not vel_dir.exists() or not lab_dir.exists():
        return []
    vel_ids = {int(f.stem) for f in vel_dir.glob('*.bin')}
    lab_ids = {int(f.stem) for f in lab_dir.glob('*.label')}
    all_ids = sorted(vel_ids & lab_ids)
    if n_frames:
        # Seleccionar frames distribuidos uniformemente
        stride = max(1, len(all_ids) // n_frames)
        all_ids = all_ids[::stride][:n_frames]
    return all_ids


def test_sequence(seq, scan_ids, gradient_thresholds):
    """Evalúa las 3 configuraciones en una secuencia."""

    print(f"\n{'='*100}")
    print(f"Secuencia {seq}: {len(scan_ids)} frames")
    print(f"{'='*100}")

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        wall_rejection_slope=0.9,
        wall_height_diff_threshold=0.2,
        wall_kdtree_radius=0.3,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    # Acumuladores por configuración
    configs = {}

    # Config 1: baseline (Stage 1 solo)
    configs['PW++ + WR (baseline)'] = {'tp': 0, 'fp': 0, 'fn': 0, 'time_ms': 0}

    # Config 2: delta-r clásico
    configs['PW++ + WR + delta-r'] = {'tp': 0, 'fp': 0, 'fn': 0, 'time_ms': 0}

    # Config 3: ring gradient (varios thresholds)
    for thr in gradient_thresholds:
        configs[f'PW++ + WR + ring-grad (thr={thr})'] = {'tp': 0, 'fp': 0, 'fn': 0, 'time_ms': 0}

    # Config 4: ring gradient solo rescate (varios thresholds)
    for thr in gradient_thresholds:
        configs[f'PW++ + WR + ring-grad-rescue (thr={thr})'] = {'tp': 0, 'fp': 0, 'fn': 0, 'time_ms': 0}

    for i, scan_id in enumerate(scan_ids):
        pts, gt_mask, valid_mask = load_scan(scan_id, seq)

        # === Stage 1: PW++ + WR ===
        t0 = time.time()
        s1 = pipeline.stage1_complete(pts)
        t_s1 = (time.time() - t0) * 1000

        baseline_mask = np.zeros(len(pts), dtype=bool)
        baseline_mask[s1['nonground_indices']] = True

        # Acumular baseline
        gt_v = gt_mask & valid_mask
        pred_v = baseline_mask & valid_mask
        tp = int(np.sum(gt_v & pred_v))
        fp = int(np.sum((~gt_v) & pred_v))
        fn = int(np.sum(gt_v & (~pred_v)))
        configs['PW++ + WR (baseline)']['tp'] += tp
        configs['PW++ + WR (baseline)']['fp'] += fp
        configs['PW++ + WR (baseline)']['fn'] += fn
        configs['PW++ + WR (baseline)']['time_ms'] += t_s1

        # === Stage 2: delta-r clásico ===
        t0 = time.time()
        s2 = pipeline.stage2_complete(pts)
        t_s2 = (time.time() - t0) * 1000

        pred_v2 = s2['obs_mask'] & valid_mask
        tp2 = int(np.sum(gt_v & pred_v2))
        fp2 = int(np.sum((~gt_v) & pred_v2))
        fn2 = int(np.sum(gt_v & (~pred_v2)))
        configs['PW++ + WR + delta-r']['tp'] += tp2
        configs['PW++ + WR + delta-r']['fp'] += fp2
        configs['PW++ + WR + delta-r']['fn'] += fn2
        configs['PW++ + WR + delta-r']['time_ms'] += t_s2

        # === Ring gradient ===
        t0 = time.time()
        ring_ids = assign_ring_ids(pts)
        t_rings = (time.time() - t0) * 1000

        for thr in gradient_thresholds:
            t0 = time.time()
            border_mask = compute_ring_gradient_vectorized(pts, ring_ids, thr)
            t_grad = (time.time() - t0) * 1000

            # --- Modo libre: ring gradient reemplaza delta-r ---
            # non-ground de Stage 1 + bordes detectados por ring gradient
            grad_mask = baseline_mask | border_mask
            pred_vg = grad_mask & valid_mask
            tp_g = int(np.sum(gt_v & pred_vg))
            fp_g = int(np.sum((~gt_v) & pred_vg))
            fn_g = int(np.sum(gt_v & (~pred_vg)))
            key = f'PW++ + WR + ring-grad (thr={thr})'
            configs[key]['tp'] += tp_g
            configs[key]['fp'] += fp_g
            configs[key]['fn'] += fn_g
            configs[key]['time_ms'] += t_s1 + t_rings + t_grad

            # --- Modo rescate: solo ground→nonground ---
            # Solo añade puntos que Stage 1 clasificó como ground
            # pero ring gradient detecta como borde de obstáculo
            ground_mask_s1 = ~baseline_mask
            rescue_mask = border_mask & ground_mask_s1  # solo rescata ground points
            rescue_pred = baseline_mask | rescue_mask
            pred_vr = rescue_pred & valid_mask
            tp_r = int(np.sum(gt_v & pred_vr))
            fp_r = int(np.sum((~gt_v) & pred_vr))
            fn_r = int(np.sum(gt_v & (~pred_vr)))
            key_r = f'PW++ + WR + ring-grad-rescue (thr={thr})'
            configs[key_r]['tp'] += tp_r
            configs[key_r]['fp'] += fp_r
            configs[key_r]['fn'] += fn_r
            configs[key_r]['time_ms'] += t_s1 + t_rings + t_grad

        if (i + 1) % max(1, len(scan_ids) // 10) == 0:
            print(f"\r  [{i+1}/{len(scan_ids)}] {100*(i+1)/len(scan_ids):.0f}%", end="", flush=True)

    print()

    # Tabla de resultados
    print(f"\n{'Configuración':<50} | {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8} {'ms/fr':>8}")
    print("-" * 100)

    baseline_f1 = None
    for name, acc in configs.items():
        tp, fp, fn = acc['tp'], acc['fp'], acc['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ms = acc['time_ms'] / len(scan_ids)

        delta = ""
        if baseline_f1 is not None:
            df1 = f1 - baseline_f1
            delta = f"  ({100*df1:+.2f}%)"
        else:
            baseline_f1 = f1

        print(f"{name:<50} | {100*f1:>7.2f}% {100*iou:>7.2f}% "
              f"{100*p:>7.2f}% {100*r:>7.2f}% {ms:>7.1f}{delta}")

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Test Ring Gradient Anomaly Detection (Stage 2 alternativo)')
    parser.add_argument('--seq', type=str, default='both',
                        choices=['00', '04', 'both'],
                        help='Secuencia a evaluar (default: both)')
    parser.add_argument('--n_frames', type=int, default=30,
                        help='Frames por secuencia (default: 30)')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
                        help='Umbrales de gradiente a probar (m)')
    args = parser.parse_args()

    seqs = ['00', '04'] if args.seq == 'both' else [args.seq]

    print("=" * 100)
    print("RING GRADIENT ANOMALY DETECTION — Stage 2 Alternativo")
    print("=" * 100)
    print(f"\nConcepto: detectar bordes de obstáculos por discontinuidades de rango")
    print(f"          entre puntos consecutivos del mismo ring del LiDAR.")
    print(f"          No depende de planos RANSAC — solo geometría del ring.")
    print(f"\nModos:")
    print(f"  ring-grad:        Stage 1 OR ring gradient (modo libre)")
    print(f"  ring-grad-rescue: Stage 1 + ring gradient solo rescata ground→obs")
    print(f"\nSecuencias: {seqs}")
    print(f"Frames/seq: {args.n_frames}")
    print(f"Umbrales gradient: {args.thresholds} m")

    for seq in seqs:
        scan_ids = discover_scan_ids(seq, args.n_frames)
        if not scan_ids:
            print(f"\nSeq {seq}: sin datos, saltando")
            continue
        test_sequence(seq, scan_ids, args.thresholds)


if __name__ == '__main__':
    main()
