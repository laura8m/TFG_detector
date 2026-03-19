#!/usr/bin/env python3
"""
Grid Search de parámetros del filtro Bayesiano (Stage 3).

Stage 2 solo da F1=89.9% y Stage 2→3 baja a 87.4%. Este test busca los parámetros
óptimos de Stage 3 para que MEJORE sobre el baseline de Stage 2.

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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages')

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


# ========================================
# UTILIDADES (reutilizadas de test_full_ablation.py)
# ========================================

SEQUENCES = {
    '04': {
        'data_dir': Path(__file__).parent.parent / "data_kitti" / "04" / "04",
        'label_dir': Path(__file__).parent.parent / "data_kitti" / "04_labels" / "04" / "labels",
        'poses_file': str(Path(__file__).parent.parent / "data_kitti" / "04_labels" / "04" / "poses.txt"),
    },
    '00': {
        'data_dir': Path(__file__).parent.parent / "data_kitti" / "00" / "00",
        'label_dir': Path(__file__).parent.parent / "data_kitti" / "00_labels" / "00" / "labels",
        'poses_file': str(Path(__file__).parent.parent / "data_kitti" / "00_labels" / "00" / "poses.txt"),
    }
}


def load_kitti_scan(scan_id: int, seq: str):
    info = SEQUENCES[seq]
    scan_file = info['data_dir'] / "velodyne" / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = info['label_dir'] / f"{scan_id:06d}.label"
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


def grid_to_configs(grid):
    """Genera todas las combinaciones del grid como PipelineConfig."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        config = PipelineConfig(
            # Stage 3 params (grid)
            gamma=params['gamma'],
            gamma_min=0.0,
            gamma_speed_threshold=0.8,
            gamma_speed_scale=2.0,
            belief_clamp_min=-params['belief_clamp_max'],
            belief_clamp_max=params['belief_clamp_max'],
            prob_threshold_obs=params['prob_threshold_obs'],
            depth_jump_threshold=params['depth_jump_threshold'],
            # Stage 4+5 desactivados para aislar Stage 3
            enable_shadow_validation=False,
            enable_cluster_filtering=False,
            verbose=False,
        )
        configs.append((params, config))
    return configs


# ========================================
# RUNNER
# ========================================

def run_stage2_baseline(seq, scan_start, n_frames, poses, gt_mask):
    """Ejecuta Stage 2 solo como referencia."""
    config = PipelineConfig(
        enable_shadow_validation=False,
        enable_cluster_filtering=False,
        verbose=False,
    )
    pipeline = LidarPipelineSuite(config)
    scan_ref = scan_start + n_frames - 1
    pts, _ = load_kitti_scan(scan_ref, seq)
    result = pipeline.stage2_complete(pts)
    return compute_metrics(gt_mask, result['obs_mask'])


def run_single_config(params, config, seq, scan_start, n_frames, poses, gt_mask):
    """Ejecuta Stage 2→3 con una configuración y devuelve métricas."""
    pipeline = LidarPipelineSuite(config)

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        delta_pose = None if i == 0 else LidarPipelineSuite.compute_delta_pose(
            poses[scan_start + i - 1], poses[scan_start + i]
        )
        result = pipeline.stage3_per_point(pts, delta_pose=delta_pose)

    return compute_metrics(gt_mask, result['obs_mask'])


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
    configs = grid_to_configs(grid)
    n_combos = len(configs)

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

    # Resultados por secuencia
    all_results = {}

    for seq in seqs:
        info = SEQUENCES[seq]
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

        # Baseline Stage 2
        print("\n  Ejecutando baseline Stage 2...", end=" ", flush=True)
        baseline = run_stage2_baseline(seq, args.scan_start, args.n_frames, poses, gt_mask)
        print(f"F1={100*baseline['f1']:.1f}%  P={100*baseline['precision']:.1f}%  R={100*baseline['recall']:.1f}%")

        # Grid search
        results = []
        t0_total = time.time()

        for idx, (params, config) in enumerate(configs):
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - t0_total
                eta = (elapsed / max(idx, 1)) * (n_combos - idx)
                print(f"\r  Probando combinación {idx+1}/{n_combos} (ETA: {eta:.0f}s)...", end="", flush=True)

            t0 = time.time()
            metrics = run_single_config(params, config, seq, args.scan_start, args.n_frames, poses, gt_mask)
            elapsed = time.time() - t0

            results.append({
                'params': params,
                'metrics': metrics,
                'delta_f1': metrics['f1'] - baseline['f1'],
                'elapsed': elapsed,
            })

        elapsed_total = time.time() - t0_total
        print(f"\r  {n_combos} combinaciones completadas en {elapsed_total:.0f}s" + " " * 30)

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

        # Cuántos mejoran sobre baseline
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

        # Construir tabla combinada: para cada combo, media F1 de ambas secuencias
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
