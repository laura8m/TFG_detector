#!/usr/bin/env python3
"""
Test Ablation Completo del Pipeline

Ablation acumulativo: cada stage se añade progresivamente para medir su contribución.
- Config 1: Stage 2 solo (single-frame baseline)
- Config 2: Stage 2 → 3 (+ Bayesian temporal con gamma adaptativo)
- Config 3: Stage 2 → 3 → 4 (+ DBSCAN Cluster Filtering) [pipeline completo]

Métricas: Precision, Recall, F1, FP, FN + timing desglosado por stage.
Secuencias: KITTI 00 (urbano) y 04 (highway).
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages')

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# ========================================
# UTILIDADES
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
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,                    # Persons
        50, 51, 52,                    # Structures
        70, 71,                        # Vegetation (NO 72=terrain)
        80, 81,                        # Poles/signs
        99,                            # other-object
        252, 253, 254, 255, 256, 257, 258, 259  # Moving objects
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


# ========================================
# ABLATION CONFIGS
# ========================================

def get_ablation_configs(include_hcd_ablation=False):
    """Retorna dict de configs para ablation acumulativo.
    Si include_hcd_ablation=True, añade variantes sin HCD para medir su impacto."""
    configs = {
        'Stage 2 (baseline)': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=True,
            enable_cluster_filtering=False,
            verbose=False
        ),
        'Stage 2→3': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=True,
            enable_cluster_filtering=False,
            verbose=False
        ),
        'Stage 2→3→4': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=True,
            enable_cluster_filtering=True,
            cluster_min_pts=15,
            verbose=False
        ),
    }

    if include_hcd_ablation:
        # Variantes SIN HCD para medir su contribución
        configs['Stage 2 (sin HCD)'] = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=False,
            enable_cluster_filtering=False,
            verbose=False
        )
        configs['Stage 2→3→4 (sin HCD)'] = PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=False,
            enable_cluster_filtering=True,
            cluster_min_pts=15,
            verbose=False
        )

    return configs


# ========================================
# RUNNER POR CONFIG
# ========================================

def run_config(config_name, config, seq, scan_start, n_frames, poses):
    """Ejecutar una config y devolver métricas + timings del último frame"""

    pipeline = LidarPipelineSuite(config)

    # Stage 2 solo: single-frame (solo el último frame)
    if config_name == 'Stage 2 (baseline)':
        scan_ref = scan_start + n_frames - 1
        pts, _ = load_kitti_scan(scan_ref, seq)
        result = pipeline.stage2_complete(pts)
        return result, {
            'timing_s12_ms': result.get('timing_total_ms', 0),
            'timing_stage3_ms': 0,
            'timing_stage4_ms': 0,
            'timing_total_ms': result.get('timing_total_ms', 0),
        }

    # Multi-frame: ejecutar todos los frames
    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        delta_pose = None if i == 0 else LidarPipelineSuite.compute_delta_pose(
            poses[scan_start + i - 1], poses[scan_start + i]
        )

        # Elegir método según config
        if config.enable_cluster_filtering:
            result = pipeline.stage4_per_point(pts, delta_pose=delta_pose)
        else:
            result = pipeline.stage3_per_point(pts, delta_pose=delta_pose)

    # Extraer timings del último frame
    # Stage 1+2 se ejecutan dentro de stage3_per_point, su timing está en timing_ms de stage2
    # timing_stage3_ms incluye stage2 internamente, así que restamos
    t_s3 = result.get('timing_stage3_ms', 0)
    t_s4 = result.get('timing_stage4_ms', 0)
    t_total = result.get('timing_total_ms', 0)
    # Stage 1+2 = total - stage3 - stage4 (stage3 ya incluye s1+s2 internamente)
    # Pero timing_stage3_ms es solo el tiempo de warp+bayes, no incluye s1+s2
    # timing_total_ms = s1+s2 + s3_warp_bayes + s4
    t_s12 = t_total - t_s3 - t_s4

    timings = {
        'timing_s12_ms': max(0, t_s12),
        'timing_stage3_ms': t_s3,
        'timing_stage4_ms': t_s4,
        'timing_total_ms': t_total,
    }

    return result, timings


# ========================================
# TEST POR SECUENCIA
# ========================================

def test_sequence(seq, scan_start, n_frames, include_hcd_ablation=False):
    info = SEQUENCES[seq]
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])

    # GT del último frame
    scan_ref = scan_start + n_frames - 1
    _, labels_ref = load_kitti_scan(scan_ref, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)

    print("=" * 90)
    print(f"SECUENCIA {seq} | Frames {scan_start}-{scan_ref} ({n_frames} frames) | GT obstacles: {gt_mask.sum()}")
    print("=" * 90)

    configs = get_ablation_configs(include_hcd_ablation=include_hcd_ablation)
    all_metrics = {}
    all_timings = {}

    for name, config in configs.items():
        print(f"\n  Ejecutando: {name}...", end=" ", flush=True)
        t0 = time.time()

        result, timings = run_config(name, config, seq, scan_start, n_frames, poses)
        metrics = compute_metrics(gt_mask, result['obs_mask'])

        all_metrics[name] = metrics
        all_timings[name] = timings

        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s)")

    # ========================================
    # TABLA DE MÉTRICAS
    # ========================================
    print(f"\n{'='*90}")
    print(f"MÉTRICAS - SECUENCIA {seq}")
    print(f"{'='*90}")
    print(f"\n{'Config':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'FP':>8} {'FN':>8}")
    print("-" * 94)
    for name, m in all_metrics.items():
        print(f"{name:<28} {100*m['precision']:>9.1f}% {100*m['recall']:>9.1f}% {100*m['f1']:>9.1f}% {100*m['iou']:>9.1f}% {m['fp']:>8} {m['fn']:>8}")

    # ========================================
    # TABLA DE TIMING
    # ========================================
    print(f"\n{'='*90}")
    print(f"TIMING (último frame) - SECUENCIA {seq}")
    print(f"{'='*90}")
    print(f"\n{'Config':<28} {'S1+S2':>8} {'S3':>8} {'S4':>8} {'Total':>8}")
    print("-" * 68)
    for name, t in all_timings.items():
        print(f"{name:<28} {t['timing_s12_ms']:>7.0f}ms {t['timing_stage3_ms']:>7.0f}ms {t['timing_stage4_ms']:>7.0f}ms {t['timing_total_ms']:>7.0f}ms")

    # ========================================
    # CONTRIBUCIÓN DE CADA STAGE
    # ========================================
    print(f"\n{'='*90}")
    print(f"CONTRIBUCIÓN POR STAGE - SECUENCIA {seq}")
    print(f"{'='*90}")

    names = list(all_metrics.keys())
    # Stage 3 contribution: (2→3) vs (2)
    if 'Stage 2 (baseline)' in all_metrics and 'Stage 2→3' in all_metrics:
        m_base = all_metrics['Stage 2 (baseline)']
        m_s3 = all_metrics['Stage 2→3']
        print(f"\n  Stage 3 (Bayesian temporal + gamma adaptativo):")
        print(f"    F1:        {100*(m_s3['f1']-m_base['f1']):+.2f}%")
        print(f"    IoU:       {100*(m_s3['iou']-m_base['iou']):+.2f}%")
        print(f"    Precision: {100*(m_s3['precision']-m_base['precision']):+.2f}%")
        print(f"    Recall:    {100*(m_s3['recall']-m_base['recall']):+.2f}%")
        print(f"    FP:        {m_s3['fp']-m_base['fp']:+d}")

    # Stage 4 contribution: (2→3→4) vs (2→3)
    if 'Stage 2→3' in all_metrics and 'Stage 2→3→4' in all_metrics:
        m_s3 = all_metrics['Stage 2→3']
        m_s4 = all_metrics['Stage 2→3→4']
        print(f"\n  Stage 4 (DBSCAN Cluster Filtering):")
        print(f"    F1:        {100*(m_s4['f1']-m_s3['f1']):+.2f}%")
        print(f"    IoU:       {100*(m_s4['iou']-m_s3['iou']):+.2f}%")
        print(f"    Precision: {100*(m_s4['precision']-m_s3['precision']):+.2f}%")
        print(f"    Recall:    {100*(m_s4['recall']-m_s3['recall']):+.2f}%")
        print(f"    FP:        {m_s4['fp']-m_s3['fp']:+d}")

    # Pipeline completo vs baseline
    if 'Stage 2 (baseline)' in all_metrics and 'Stage 2→3→4' in all_metrics:
        m_base = all_metrics['Stage 2 (baseline)']
        m_full = all_metrics['Stage 2→3→4']
        print(f"\n  TOTAL (Pipeline completo vs Stage 2 solo):")
        print(f"    F1:        {100*(m_full['f1']-m_base['f1']):+.2f}%")
        print(f"    IoU:       {100*(m_full['iou']-m_base['iou']):+.2f}%")
        print(f"    Precision: {100*(m_full['precision']-m_base['precision']):+.2f}%")
        print(f"    Recall:    {100*(m_full['recall']-m_base['recall']):+.2f}%")

    # HCD ablation: comparar con y sin HCD
    if 'Stage 2 (sin HCD)' in all_metrics and 'Stage 2 (baseline)' in all_metrics:
        print(f"\n{'='*90}")
        print(f"ABLATION HCD - SECUENCIA {seq}")
        print(f"{'='*90}")

        m_hcd = all_metrics['Stage 2 (baseline)']
        m_nohcd = all_metrics['Stage 2 (sin HCD)']
        print(f"\n  Stage 2 single-frame:")
        print(f"    Con HCD:   F1={100*m_hcd['f1']:.1f}%  IoU={100*m_hcd['iou']:.1f}%  P={100*m_hcd['precision']:.1f}%  R={100*m_hcd['recall']:.1f}%  FP={m_hcd['fp']}")
        print(f"    Sin HCD:   F1={100*m_nohcd['f1']:.1f}%  IoU={100*m_nohcd['iou']:.1f}%  P={100*m_nohcd['precision']:.1f}%  R={100*m_nohcd['recall']:.1f}%  FP={m_nohcd['fp']}")
        print(f"    Delta:     F1={100*(m_hcd['f1']-m_nohcd['f1']):+.2f}%  IoU={100*(m_hcd['iou']-m_nohcd['iou']):+.2f}%  FP={m_hcd['fp']-m_nohcd['fp']:+d}")

    if 'Stage 2→3→4 (sin HCD)' in all_metrics and 'Stage 2→3→4' in all_metrics:
        m_full_hcd = all_metrics['Stage 2→3→4']
        m_full_nohcd = all_metrics['Stage 2→3→4 (sin HCD)']
        print(f"\n  Pipeline completo (Stage 2→3→4):")
        print(f"    Con HCD:   F1={100*m_full_hcd['f1']:.1f}%  IoU={100*m_full_hcd['iou']:.1f}%  P={100*m_full_hcd['precision']:.1f}%  R={100*m_full_hcd['recall']:.1f}%  FP={m_full_hcd['fp']}")
        print(f"    Sin HCD:   F1={100*m_full_nohcd['f1']:.1f}%  IoU={100*m_full_nohcd['iou']:.1f}%  P={100*m_full_nohcd['precision']:.1f}%  R={100*m_full_nohcd['recall']:.1f}%  FP={m_full_nohcd['fp']}")
        print(f"    Delta:     F1={100*(m_full_hcd['f1']-m_full_nohcd['f1']):+.2f}%  IoU={100*(m_full_hcd['iou']-m_full_nohcd['iou']):+.2f}%  FP={m_full_hcd['fp']-m_full_nohcd['fp']:+d}")

    return all_metrics, all_timings


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Ablation Study')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--hcd', action='store_true', help='Incluir ablation de HCD (con vs sin)')
    args = parser.parse_args()

    all_results = {}

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', args.scan_start, args.n_frames, args.hcd)

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', args.scan_start, args.n_frames, args.hcd)

    # Resumen global
    if len(all_results) > 1 and all(v is not None for v in all_results.values()):
        print("\n" + "=" * 90)
        print("RESUMEN GLOBAL (ambas secuencias)")
        print("=" * 90)

        config_names = list(all_results['04'][0].keys())
        print(f"\n{'Config':<28} {'Seq 04 F1':>10} {'Seq 00 F1':>10} {'Media F1':>10} {'Seq 04 IoU':>11} {'Seq 00 IoU':>11} {'Media IoU':>10}")
        print("-" * 100)
        for name in config_names:
            f1_04 = all_results['04'][0][name]['f1']
            f1_00 = all_results['00'][0][name]['f1']
            f1_avg = (f1_04 + f1_00) / 2
            iou_04 = all_results['04'][0][name]['iou']
            iou_00 = all_results['00'][0][name]['iou']
            iou_avg = (iou_04 + iou_00) / 2
            print(f"{name:<28} {100*f1_04:>9.1f}% {100*f1_00:>9.1f}% {100*f1_avg:>9.1f}% {100*iou_04:>10.1f}% {100*iou_00:>10.1f}% {100*iou_avg:>9.1f}%")


if __name__ == '__main__':
    main()
