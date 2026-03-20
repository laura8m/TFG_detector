#!/usr/bin/env python3
"""
Grid Search de parámetros del filtro Bayesiano (Stage 3).

Stage 2 solo da F1=89.9% y Stage 2→3 baja a 87.4%. Este test busca los parámetros
óptimos de Stage 3 para que MEJORE sobre el baseline de Stage 2.

OPTIMIZACIÓN: Precomputa Stage 1+2 y las asociaciones KDTree (warping) una sola vez.
Solo replaya la actualización Bayesiana (aritmética NumPy pura) por cada combinación
del grid. Esto reduce el tiempo de ~360x pipeline completo a ~1x pipeline + 360x NumPy.

Parámetros explorados:
- gamma: inercia del belief previo (0 = single-frame, 1 = solo prior)
- belief_clamp: saturación del log-odds (evita acumulación infinita)
- prob_threshold_obs: umbral de probabilidad para clasificar obstáculo
- depth_jump_threshold: umbral para invalidar asociaciones por salto de profundidad

Uso:
    python3 tests/test_bayes_grid_search.py --seq both --n_frames 10
    python3 tests/test_bayes_grid_search.py --seq 04 --n_frames 10 --top 20
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time
from itertools import product
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file


# ========================================
# UTILIDADES
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


def compute_metrics(gt_mask, pred_mask):
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def get_gt_obstacle_mask(semantic_labels):
    """SemanticKITTI obstacle labels (NO 72=terrain, SI 252-259=moving)"""
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,
        30, 31, 32,
        50, 51, 52,
        70, 71,
        80, 81,
        99,
        252, 253, 254, 255, 256, 257, 258, 259
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


# ========================================
# GRID DE PARÁMETROS
# ========================================

def get_parameter_grid():
    """Define el grid de parámetros a explorar."""
    grid = {
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
        'belief_clamp_max': [2.0, 3.0, 5.0, 10.0],
        'prob_threshold_obs': [0.35, 0.45, 0.5, 0.55, 0.6],
        'depth_jump_threshold': [1.0, 2.0, 3.0],
    }
    return grid


# ========================================
# PRECOMPUTACIÓN (1 sola vez por secuencia)
# ========================================

def precompute_stage2_and_warp(seq, scan_start, n_frames, poses):
    """
    Ejecuta Stage 1+2 y calcula las asociaciones KDTree para todos los frames.

    Retorna por frame:
    - likelihood: (N,) log-odds de Stage 2
    - points: (N, 3)
    - warp_distances: (N,) distancia al punto más cercano del frame anterior
    - warp_indices: (N,) índice del punto más cercano del frame anterior
    - ego_speed: velocidad del ego-vehículo (m/frame)
    - ranges_current: (N,) rango de cada punto actual
    - ranges_prev_warped: (N,) rango del punto asociado del frame anterior (para depth jump)
    """
    config = PipelineConfig(
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)

    frames_data = []

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)

        # Stage 1+2
        stage2_result = pipeline.stage2_complete(pts)
        likelihood = stage2_result['likelihood']

        # Calcular delta_pose y ego_speed
        delta_pose = None
        ego_speed = 0.0
        if i > 0:
            delta_pose = LidarPipelineSuite.compute_delta_pose(
                poses[scan_start + i - 1], poses[scan_start + i]
            )
            ego_speed = np.linalg.norm(delta_pose[:3, 3])

        # Calcular asociaciones KDTree (warping)
        warp_distances = None
        warp_indices = None
        ranges_current = None
        ranges_prev_warped = None

        if i > 0:
            prev_data = frames_data[i - 1]
            pts_prev = prev_data['points']

            # Transformar puntos del frame anterior al actual
            if delta_pose is not None:
                pts_prev_hom = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
                pts_prev_warped = (delta_pose @ pts_prev_hom.T).T[:, :3]
            else:
                pts_prev_warped = pts_prev

            # KDTree: asociar cada punto actual con el más cercano del anterior
            tree = cKDTree(pts_prev_warped)
            max_distance = 2.0
            warp_distances, warp_indices = tree.query(pts, k=1, distance_upper_bound=max_distance, workers=-1)

            # Rangos para depth jump check
            ranges_current = np.sqrt(np.sum(pts**2, axis=1))
            ranges_prev_warped = np.sqrt(np.sum(pts_prev_warped**2, axis=1))

        frames_data.append({
            'points': pts,
            'likelihood': likelihood,
            'ego_speed': ego_speed,
            'warp_distances': warp_distances,
            'warp_indices': warp_indices,
            'ranges_current': ranges_current,
            'ranges_prev_warped': ranges_prev_warped,
        })

    return frames_data


# ========================================
# REPLAY BAYESIANO (solo aritmética NumPy)
# ========================================

def replay_bayes(frames_data, params):
    """
    Replaya la actualización Bayesiana con parámetros dados.
    Solo hace aritmética NumPy sobre datos precomputados → ~1ms por frame.
    """
    gamma_base = params['gamma']
    gamma_min = 0.0
    gamma_speed_threshold = 0.8
    gamma_speed_scale = 2.0
    clamp_max = params['belief_clamp_max']
    clamp_min = -clamp_max
    prob_thr = params['prob_threshold_obs']
    depth_jump_thr = params['depth_jump_threshold']
    warp_min_association = 0.3
    l0 = 0.0

    belief_prev = None
    n_frames = len(frames_data)

    for i in range(n_frames):
        fd = frames_data[i]
        likelihood = fd['likelihood']
        N = len(likelihood)

        if i == 0 or belief_prev is None:
            # Primer frame: belief = likelihood (sin temporal)
            belief_warped = np.full(N, l0, dtype=np.float32)
        else:
            # Warping: heredar belief del frame anterior usando asociaciones precomputadas
            belief_warped = np.full(N, l0, dtype=np.float32)
            distances = fd['warp_distances']
            indices = fd['warp_indices']

            valid_mask = distances < 2.0  # max_distance

            # Depth jump check con parámetro del grid
            if depth_jump_thr > 0 and np.any(valid_mask):
                valid_indices_arr = np.where(valid_mask)[0]
                r_curr = fd['ranges_current'][valid_mask]
                r_prev = fd['ranges_prev_warped'][indices[valid_mask]]
                depth_jump = np.abs(r_curr - r_prev) > depth_jump_thr
                valid_mask[valid_indices_arr[depth_jump]] = False

            # Heredar belief
            belief_warped[valid_mask] = belief_prev[indices[valid_mask]]

            n_associated = valid_mask.sum()
            association_ratio = n_associated / max(N, 1)

        # Gamma adaptativo por velocidad
        gamma = gamma_base
        ego_speed = fd['ego_speed']

        if ego_speed > gamma_speed_threshold:
            t_interp = min(1.0, (ego_speed - gamma_speed_threshold) /
                           (gamma_speed_scale - gamma_speed_threshold))
            gamma = gamma_base - t_interp * (gamma_base - gamma_min)

        # Protección: warping inestable
        if i > 0:
            if association_ratio < warp_min_association and association_ratio > 0:
                gamma = gamma_min

        # Actualización Bayesiana
        belief = likelihood + gamma * (belief_warped - l0) + l0
        belief = np.clip(belief, clamp_min, clamp_max)

        # Guardar para próximo frame
        belief_prev = belief.copy()

    # Umbral final (último frame)
    belief_prob = 1.0 / (1.0 + np.exp(-belief))
    obs_mask = belief_prob > prob_thr

    return obs_mask


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Grid Search de parámetros Bayesianos (Stage 3)')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--top', type=int, default=15, help='Mostrar los N mejores resultados')
    args = parser.parse_args()

    grid = get_parameter_grid()
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print("=" * 100)
    print("GRID SEARCH - PARÁMETROS BAYESIANOS (STAGE 3)")
    print("=" * 100)
    print(f"\nParámetros explorados:")
    for key, values in grid.items():
        print(f"  {key}: {values}")
    print(f"\nTotal combinaciones: {n_combos}")

    seqs = []
    if args.seq in ('04', 'both'):
        seqs.append('04')
    if args.seq in ('00', 'both'):
        seqs.append('00')

    all_results = {}

    for seq in seqs:
        info = get_sequence_info(seq)
        if not info['data_dir'].exists():
            print(f"\n  [SKIP] Datos no encontrados: {info['data_dir']}")
            continue

        poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])
        scan_ref = args.scan_start + args.n_frames - 1
        _, labels_ref = load_kitti_scan(scan_ref, seq)
        gt_mask = get_gt_obstacle_mask(labels_ref)

        print(f"\n{'='*100}")
        print(f"SECUENCIA {seq} | Frames {args.scan_start}-{scan_ref} ({args.n_frames} frames) | GT obstacles: {gt_mask.sum()}")
        print(f"{'='*100}")

        # ========================================
        # FASE 1: Precomputar Stage 1+2 + KDTree (1 sola vez)
        # ========================================
        print("\n  [Precomputo] Stage 1+2 + asociaciones KDTree...", end=" ", flush=True)
        t0_pre = time.time()
        frames_data = precompute_stage2_and_warp(seq, args.scan_start, args.n_frames, poses)
        t_pre = time.time() - t0_pre
        print(f"OK ({t_pre:.1f}s)")

        # Baseline Stage 2 (último frame, sin temporal)
        baseline_obs = frames_data[-1]['likelihood'] > 0  # likelihood > 0 → Stage 2 dice obstáculo
        # Más preciso: usar el pipeline para Stage 2 baseline
        config_baseline = PipelineConfig(enable_cluster_filtering=False, verbose=False)
        pipeline_baseline = LidarPipelineSuite(config_baseline)
        pts_ref, _ = load_kitti_scan(scan_ref, seq)
        result_baseline = pipeline_baseline.stage2_complete(pts_ref)
        baseline = compute_metrics(gt_mask, result_baseline['obs_mask'])
        print(f"  Baseline Stage 2: F1={100*baseline['f1']:.1f}%  P={100*baseline['precision']:.1f}%  R={100*baseline['recall']:.1f}%")

        # ========================================
        # FASE 2: Replay Bayesiano para cada combinación (solo NumPy)
        # ========================================
        keys = list(grid.keys())
        values = list(grid.values())
        results = []
        t0_grid = time.time()

        for idx, combo in enumerate(product(*values)):
            params = dict(zip(keys, combo))

            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t0_grid
                eta = (elapsed / max(idx, 1)) * (n_combos - idx)
                print(f"\r  [Replay Bayes] {idx+1}/{n_combos} (ETA: {eta:.0f}s)...", end="", flush=True)

            obs_mask = replay_bayes(frames_data, params)
            metrics = compute_metrics(gt_mask, obs_mask)

            results.append({
                'params': params,
                'metrics': metrics,
                'delta_f1': metrics['f1'] - baseline['f1'],
            })

        t_grid = time.time() - t0_grid
        print(f"\r  [Replay Bayes] {n_combos} combinaciones en {t_grid:.1f}s ({1000*t_grid/n_combos:.1f}ms/combo)" + " " * 20)

        # Ordenar por F1 descendente
        results.sort(key=lambda x: x['metrics']['f1'], reverse=True)
        all_results[seq] = {'baseline': baseline, 'results': results}

        # Top N
        print(f"\n  {'='*96}")
        print(f"  TOP {args.top} - SECUENCIA {seq} (Baseline F1={100*baseline['f1']:.1f}%)")
        print(f"  {'='*96}")
        print(f"  {'#':>3} {'gamma':>6} {'clamp':>6} {'p_thr':>6} {'d_jump':>6} | {'F1':>7} {'dF1':>7} {'P':>7} {'R':>7} {'FP':>7} {'FN':>7}")
        print(f"  {'-'*90}")

        for i, r in enumerate(results[:args.top]):
            p = r['params']
            m = r['metrics']
            marker = " *" if r['delta_f1'] > 0 else ""
            print(f"  {i+1:>3} {p['gamma']:>6.2f} {p['belief_clamp_max']:>6.1f} {p['prob_threshold_obs']:>6.2f} {p['depth_jump_threshold']:>6.1f}"
                  f" | {100*m['f1']:>6.1f}% {100*r['delta_f1']:>+6.2f}% {100*m['precision']:>6.1f}% {100*m['recall']:>6.1f}% {m['fp']:>7} {m['fn']:>7}{marker}")

        # Resumen
        n_better = sum(1 for r in results if r['delta_f1'] > 0)
        n_worse = sum(1 for r in results if r['delta_f1'] < 0)
        n_equal = sum(1 for r in results if r['delta_f1'] == 0)
        print(f"\n  Resumen: {n_better} mejoran, {n_worse} empeoran, {n_equal} iguales vs baseline")

        if n_better > 0:
            best = results[0]
            print(f"  MEJOR: gamma={best['params']['gamma']}, clamp={best['params']['belief_clamp_max']}, "
                  f"p_thr={best['params']['prob_threshold_obs']}, d_jump={best['params']['depth_jump_threshold']} "
                  f"→ F1={100*best['metrics']['f1']:.1f}% ({100*best['delta_f1']:+.2f}% vs baseline)")

    # ========================================
    # RESUMEN GLOBAL (si ambas secuencias)
    # ========================================
    if len(all_results) == 2 and all(v is not None for v in all_results.values()):
        print(f"\n{'='*100}")
        print("RESUMEN GLOBAL - MEJOR COMBINACIÓN POR MEDIA F1")
        print(f"{'='*100}")

        combined = []
        results_04 = {str(r['params']): r for r in all_results['04']['results']}
        results_00 = {str(r['params']): r for r in all_results['00']['results']}

        baseline_04 = all_results['04']['baseline']
        baseline_00 = all_results['00']['baseline']
        baseline_avg = (baseline_04['f1'] + baseline_00['f1']) / 2

        for key in results_04:
            if key in results_00:
                r04 = results_04[key]
                r00 = results_00[key]
                avg_f1 = (r04['metrics']['f1'] + r00['metrics']['f1']) / 2
                combined.append({
                    'params': r04['params'],
                    'f1_04': r04['metrics']['f1'],
                    'f1_00': r00['metrics']['f1'],
                    'avg_f1': avg_f1,
                    'delta_avg': avg_f1 - baseline_avg,
                    'p_04': r04['metrics']['precision'],
                    'r_04': r04['metrics']['recall'],
                    'p_00': r00['metrics']['precision'],
                    'r_00': r00['metrics']['recall'],
                    'fp_04': r04['metrics']['fp'],
                    'fp_00': r00['metrics']['fp'],
                })

        combined.sort(key=lambda x: x['avg_f1'], reverse=True)

        print(f"\n  Baseline: Seq 04 F1={100*baseline_04['f1']:.1f}%  Seq 00 F1={100*baseline_00['f1']:.1f}%  Media={100*baseline_avg:.1f}%")
        print(f"\n  {'#':>3} {'gamma':>6} {'clamp':>6} {'p_thr':>6} {'d_jump':>6} | {'04 F1':>7} {'00 F1':>7} {'Media':>7} {'dMedia':>7} | {'04 FP':>7} {'00 FP':>7}")
        print(f"  {'-'*100}")

        for i, c in enumerate(combined[:args.top]):
            p = c['params']
            marker = " *" if c['delta_avg'] > 0 else ""
            print(f"  {i+1:>3} {p['gamma']:>6.2f} {p['belief_clamp_max']:>6.1f} {p['prob_threshold_obs']:>6.2f} {p['depth_jump_threshold']:>6.1f}"
                  f" | {100*c['f1_04']:>6.1f}% {100*c['f1_00']:>6.1f}% {100*c['avg_f1']:>6.1f}% {100*c['delta_avg']:>+6.2f}%"
                  f" | {c['fp_04']:>7} {c['fp_00']:>7}{marker}")

        n_better = sum(1 for c in combined if c['delta_avg'] > 0)
        print(f"\n  {n_better}/{len(combined)} combinaciones mejoran la media F1 sobre baseline")

        if combined and combined[0]['delta_avg'] > 0:
            best = combined[0]
            print(f"\n  ÓPTIMO GLOBAL:")
            print(f"    gamma={best['params']['gamma']}, belief_clamp={best['params']['belief_clamp_max']}, "
                  f"prob_threshold={best['params']['prob_threshold_obs']}, depth_jump={best['params']['depth_jump_threshold']}")
            print(f"    Seq 04: F1={100*best['f1_04']:.1f}%  Seq 00: F1={100*best['f1_00']:.1f}%  Media: {100*best['avg_f1']:.1f}% ({100*best['delta_avg']:+.2f}%)")
        else:
            print(f"\n  NINGUNA combinación mejora sobre el baseline de Stage 2.")
            print(f"  El filtro temporal Bayesiano no aporta con estos rangos de parámetros.")


if __name__ == '__main__':
    main()
