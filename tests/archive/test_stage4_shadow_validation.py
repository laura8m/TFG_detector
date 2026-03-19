#!/usr/bin/env python3
"""
Test Stage 4: Shadow Validation per-point 3D

Compara en ambas secuencias KITTI (00 y 04):
- Stage 2 solo (baseline)
- Stage 3 per-point con egomotion (problema de precision)
- Stage 4 = Stage 3 + Shadow Validation (¿mejora precision?)

Objetivo: Validar que Stage 4 reduce FP manteniendo recall.
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

# Add paths
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


def load_kitti_scan(scan_id: int, seq: str = '04'):
    """Cargar scan de KITTI con labels"""
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


def compute_detection_metrics(gt_mask, pred_mask):
    """Precision, Recall, F1"""
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }


def get_gt_obstacle_mask(semantic_labels):
    """Máscara de obstáculos según SemanticKITTI semantic-kitti.yaml"""
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles (car, bicycle, bus, motorcycle, on-rails, truck, other-vehicle)
        30, 31, 32,                    # Persons (person, bicyclist, motorcyclist)
        50, 51, 52,                    # Structures (building, fence, other-structure)
        70, 71,                        # Vegetation (vegetation, trunk) — NO 72 (terrain = ground)
        80, 81,                        # Poles/signs (pole, traffic-sign)
        99,                            # other-object
        252, 253, 254, 255, 256, 257, 258, 259  # Moving (car, person, bicyclist, etc.)
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


def print_metrics(name, metrics, timing_ms=None):
    """Imprimir métricas formateadas"""
    print(f"  {name}:")
    print(f"    Precision: {100*metrics['precision']:.2f}%")
    print(f"    Recall:    {100*metrics['recall']:.2f}%")
    print(f"    F1 Score:  {100*metrics['f1']:.2f}%")
    print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    if timing_ms is not None:
        print(f"    Timing: {timing_ms:.0f} ms")


# ========================================
# TEST POR SECUENCIA
# ========================================

def test_sequence(seq: str, scan_start: int, n_frames: int):
    """Ejecutar test completo en una secuencia"""
    info = SEQUENCES[seq]

    print("=" * 80)
    print(f"SECUENCIA {seq} | Frames {scan_start}-{scan_start + n_frames - 1}")
    print("=" * 80)

    # Verificar que existen los datos
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    # Cargar poses
    poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])
    print(f"  Poses cargadas: {len(poses)}")

    # Frame de referencia (último)
    scan_ref = scan_start + n_frames - 1
    points_ref, labels_ref = load_kitti_scan(scan_ref, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)
    print(f"  Frame referencia (scan {scan_ref}): {len(points_ref)} pts, {gt_mask.sum()} obstacles GT")
    print()

    results = {}

    # ========================================
    # 1. STAGE 2 SOLO (Baseline single-frame)
    # ========================================
    print("-" * 60)
    print("CONFIG 1: Stage 2 solo (baseline)")
    print("-" * 60)

    pipeline_s2 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    ))
    result_s2 = pipeline_s2.stage2_complete(points_ref)
    metrics_s2 = compute_detection_metrics(gt_mask, result_s2['obs_mask'])
    print_metrics("Stage 2", metrics_s2, result_s2['timing_total_ms'])
    results['stage2'] = metrics_s2
    print()

    # ========================================
    # 2. STAGE 3 CON EGOMOTION (problema de precision)
    # ========================================
    print("-" * 60)
    print(f"CONFIG 2: Stage 3 + egomotion ({n_frames} frames)")
    print("-" * 60)

    pipeline_s3 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    ))

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        delta_pose = None if i == 0 else LidarPipelineSuite.compute_delta_pose(
            poses[scan_start + i - 1], poses[scan_start + i]
        )
        result_s3 = pipeline_s3.stage3_per_point(pts, delta_pose=delta_pose)
        if (i + 1) % 5 == 0:
            print(f"  Frame {scan_id}: {result_s3['obs_mask'].sum()} obs")

    metrics_s3 = compute_detection_metrics(gt_mask, result_s3['obs_mask'])
    print_metrics("Stage 3+ego", metrics_s3, result_s3.get('timing_total_ms'))
    results['stage3_ego'] = metrics_s3
    print()

    # ========================================
    # 3. STAGE 4 = Stage 3 + Shadow Validation
    # ========================================
    print("-" * 60)
    print(f"CONFIG 3: Stage 4 = Stage 3 + Shadow ({n_frames} frames)")
    print("-" * 60)

    pipeline_s4 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        verbose=False
    ))

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        delta_pose = None if i == 0 else LidarPipelineSuite.compute_delta_pose(
            poses[scan_start + i - 1], poses[scan_start + i]
        )
        result_s4 = pipeline_s4.stage4_per_point(pts, delta_pose=delta_pose)
        if (i + 1) % 5 == 0:
            n_solid = result_s4.get('n_shadow_solid', 0)
            n_trans = result_s4.get('n_shadow_transparent', 0)
            n_removed = result_s4.get('n_shadow_removed', 0)
            print(f"  Frame {scan_id}: {result_s4['obs_mask'].sum()} obs | shadow: {n_solid} solid, {n_trans} transparent, {n_removed} removed")

    metrics_s4 = compute_detection_metrics(gt_mask, result_s4['obs_mask'])
    print_metrics("Stage 4", metrics_s4, result_s4.get('timing_total_ms'))
    print(f"    Stage 4 timing: {result_s4.get('timing_stage4_ms', 0):.1f} ms")
    results['stage4'] = metrics_s4
    print()

    # ========================================
    # 4. STAGE 4 sin egomotion (ablation: shadow sobre single-frame acumulado)
    # ========================================
    print("-" * 60)
    print(f"CONFIG 4: Stage 4 SIN egomotion ({n_frames} frames)")
    print("-" * 60)

    pipeline_s4_noego = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        verbose=False
    ))

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        # SIN delta_pose
        result_s4_noego = pipeline_s4_noego.stage4_per_point(pts, delta_pose=None)

    metrics_s4_noego = compute_detection_metrics(gt_mask, result_s4_noego['obs_mask'])
    print_metrics("Stage 4 (no ego)", metrics_s4_noego, result_s4_noego.get('timing_total_ms'))
    results['stage4_noego'] = metrics_s4_noego
    print()

    # ========================================
    # COMPARATIVA
    # ========================================
    print("=" * 60)
    print(f"COMPARATIVA SECUENCIA {seq}")
    print("=" * 60)
    print()
    print(f"{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8}")
    print("-" * 71)
    for name, m in results.items():
        print(f"{name:<25} {100*m['precision']:>9.2f}% {100*m['recall']:>9.2f}% {100*m['f1']:>9.2f}% {m['fp']:>8} {m['fn']:>8}")
    print()

    # Impacto de Shadow Validation
    if 'stage3_ego' in results and 'stage4' in results:
        fp_reduction = results['stage3_ego']['fp'] - results['stage4']['fp']
        fp_pct = 100 * fp_reduction / max(results['stage3_ego']['fp'], 1)
        recall_loss = 100 * (results['stage3_ego']['recall'] - results['stage4']['recall'])
        precision_gain = 100 * (results['stage4']['precision'] - results['stage3_ego']['precision'])
        f1_change = 100 * (results['stage4']['f1'] - results['stage3_ego']['f1'])

        print(f"  IMPACTO Shadow Validation (Stage 3+ego → Stage 4):")
        print(f"    FP eliminados: {fp_reduction} ({fp_pct:.1f}%)")
        print(f"    Precision:  {precision_gain:+.2f}%")
        print(f"    Recall:     {recall_loss:+.2f}% (negativo = pérdida)")
        print(f"    F1 Score:   {f1_change:+.2f}%")
        print()

    return results


# ========================================
# MAIN
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Test Stage 4 Shadow Validation')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    args = parser.parse_args()

    all_results = {}

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', args.scan_start, args.n_frames)

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', args.scan_start, args.n_frames)

    # Resumen global
    if len(all_results) > 1:
        print()
        print("=" * 80)
        print("RESUMEN GLOBAL")
        print("=" * 80)
        for seq, res in all_results.items():
            if res is None:
                continue
            print(f"\n  Secuencia {seq}:")
            print(f"    Stage 2:        F1={100*res['stage2']['f1']:.1f}%  P={100*res['stage2']['precision']:.1f}%  R={100*res['stage2']['recall']:.1f}%")
            print(f"    Stage 3+ego:    F1={100*res['stage3_ego']['f1']:.1f}%  P={100*res['stage3_ego']['precision']:.1f}%  R={100*res['stage3_ego']['recall']:.1f}%")
            print(f"    Stage 4:        F1={100*res['stage4']['f1']:.1f}%  P={100*res['stage4']['precision']:.1f}%  R={100*res['stage4']['recall']:.1f}%")
            print(f"    Stage 4 noego:  F1={100*res['stage4_noego']['f1']:.1f}%  P={100*res['stage4_noego']['precision']:.1f}%  R={100*res['stage4_noego']['recall']:.1f}%")


if __name__ == '__main__':
    main()
