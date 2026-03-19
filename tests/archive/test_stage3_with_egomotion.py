#!/usr/bin/env python3
"""
Test Stage 3 Per-Point con Egomotion Compensation

Compara:
- Stage 2 solo (baseline)
- Stage 3 per-point SIN egomotion
- Stage 3 per-point CON egomotion

Objetivo: Validar mejora de recall con egomotion compensation
"""

import sys
import numpy as np
from pathlib import Path
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages')

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# ========================================
# UTILIDADES
# ========================================

def load_kitti_scan(scan_id: int, data_dir: str = None):
    """Cargar scan de KITTI con labels"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data_kitti" / "04" / "04"

    data_dir = Path(data_dir)

    # Cargar puntos
    scan_file = data_dir / "velodyne" / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]

    # Cargar labels
    label_file = data_dir.parent.parent / "04_labels" / "04" / "labels" / f"{scan_id:06d}.label"
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)

    return points, semantic_labels


def compute_detection_metrics(gt_mask, pred_mask):
    """
    Calcular Precision, Recall, F1 para detección binaria

    Args:
        gt_mask: (N,) bool array, True = obstacle en ground truth
        pred_mask: (N,) bool array, True = obstacle detectado

    Returns:
        dict con precision, recall, f1, tp, fp, fn
    """
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def get_gt_obstacle_mask(semantic_labels):
    """
    Crear máscara de obstáculos según SemanticKITTI semantic-kitti.yaml

    Labels considerados como obstacles:
    - 10-20: Vehicles (car, bicycle, bus, motorcycle, on-rails, truck, other-vehicle)
    - 30-32: Persons (person, bicyclist, motorcyclist)
    - 50-52: Structures (building, fence, other-structure)
    - 70-71: Vegetation (vegetation, trunk) — NO 72 (terrain = ground)
    - 80-81: Poles/signs (pole, traffic-sign)
    - 99: other-object
    - 252-259: Moving objects
    """
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,                    # Persons
        50, 51, 52,                    # Structures
        70, 71,                        # Vegetation — NO 72 (terrain = ground)
        80, 81,                        # Poles/signs
        99,                            # other-object
        252, 253, 254, 255, 256, 257, 258, 259  # Moving
    ]

    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)

    return mask


# ========================================
# MAIN TEST
# ========================================

def main():
    parser = argparse.ArgumentParser(description='Test Stage 3 with Egomotion')
    parser.add_argument('--scan_start', type=int, default=0, help='Starting scan ID')
    parser.add_argument('--n_frames', type=int, default=20, help='Number of frames to process')
    parser.add_argument('--poses_file', type=str,
                       default='/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/poses.txt',
                       help='Path to KITTI poses.txt')
    args = parser.parse_args()

    print("=" * 80)
    print("TEST: STAGE 3 PER-POINT CON EGOMOTION COMPENSATION")
    print("=" * 80)
    print(f"Scan range: {args.scan_start} - {args.scan_start + args.n_frames - 1}")
    print(f"Frames to process: {args.n_frames}")
    print()

    # Cargar poses de KITTI
    print("✓ Cargando poses de KITTI...")
    poses = LidarPipelineSuite.load_kitti_poses(args.poses_file)
    print(f"  Poses cargadas: {len(poses)}")
    print()

    # Cargar último frame para referencia
    scan_id_ref = args.scan_start + args.n_frames - 1
    points_ref, semantic_labels_ref = load_kitti_scan(scan_id_ref)
    gt_mask_ref = get_gt_obstacle_mask(semantic_labels_ref)

    print(f"✓ Último frame (scan {scan_id_ref}): {len(points_ref)} puntos")
    print(f"  Ground truth obstacles: {np.sum(gt_mask_ref)}")
    print()

    # ========================================
    # CONFIG 1: STAGE 2 SOLO (Baseline)
    # ========================================
    print("-" * 80)
    print(f"CONFIG 1: STAGE 2 SOLO (Baseline - frame {scan_id_ref})")
    print("-" * 80)

    config_stage2 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    )

    pipeline_stage2 = LidarPipelineSuite(config_stage2)
    result_stage2 = pipeline_stage2.stage2_complete(points_ref)

    metrics_stage2 = compute_detection_metrics(
        gt_mask=gt_mask_ref,
        pred_mask=result_stage2['obs_mask']
    )

    print(f"✓ Obstacles detectados: {np.sum(result_stage2['obs_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {100*metrics_stage2['precision']:.2f}%")
    print(f"    Recall:    {100*metrics_stage2['recall']:.2f}%")
    print(f"    F1 Score:  {100*metrics_stage2['f1']:.2f}%")
    print(f"    TP: {metrics_stage2['tp']}, FP: {metrics_stage2['fp']}, FN: {metrics_stage2['fn']}")
    print()
    print(f"  Timing: {result_stage2['timing_total_ms']:.0f} ms")
    print()

    # ========================================
    # CONFIG 2: STAGE 3 PER-POINT SIN EGOMOTION
    # ========================================
    print("-" * 80)
    print(f"CONFIG 2: STAGE 3 PER-POINT SIN EGOMOTION ({args.n_frames} frames)")
    print("-" * 80)

    config_stage3_no_ego = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=True
    )

    pipeline_no_ego = LidarPipelineSuite(config_stage3_no_ego)

    # Procesar frames sin egomotion
    result_no_ego_final = None
    timing_no_ego_total = 0.0

    for i in range(args.n_frames):
        scan_id = args.scan_start + i
        points, _ = load_kitti_scan(scan_id)

        # SIN delta_pose
        result_no_ego = pipeline_no_ego.stage3_per_point(points, delta_pose=None)
        timing_no_ego_total += result_no_ego['timing_total_ms']

        if i == args.n_frames - 1:
            result_no_ego_final = result_no_ego

        if i % 5 == 0:
            obs_count = np.sum(result_no_ego['obs_mask'])
            print(f"  Frame {scan_id}: {obs_count} obstacles detectados")

    metrics_no_ego = compute_detection_metrics(
        gt_mask=gt_mask_ref,
        pred_mask=result_no_ego_final['obs_mask']
    )

    print()
    print(f"✓ Obstacles detectados (último frame): {np.sum(result_no_ego_final['obs_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {100*metrics_no_ego['precision']:.2f}%")
    print(f"    Recall:    {100*metrics_no_ego['recall']:.2f}%")
    print(f"    F1 Score:  {100*metrics_no_ego['f1']:.2f}%")
    print(f"    TP: {metrics_no_ego['tp']}, FP: {metrics_no_ego['fp']}, FN: {metrics_no_ego['fn']}")
    print()
    print(f"  Timing total ({args.n_frames} frames): {timing_no_ego_total:.0f} ms")
    print(f"  Timing promedio por frame: {timing_no_ego_total/args.n_frames:.0f} ms")
    print()

    # ========================================
    # CONFIG 3: STAGE 3 PER-POINT CON EGOMOTION
    # ========================================
    print("-" * 80)
    print(f"CONFIG 3: STAGE 3 PER-POINT CON EGOMOTION ({args.n_frames} frames)")
    print("-" * 80)

    config_stage3_ego = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=True
    )

    pipeline_ego = LidarPipelineSuite(config_stage3_ego)

    # Procesar frames CON egomotion
    result_ego_final = None
    timing_ego_total = 0.0

    for i in range(args.n_frames):
        scan_id = args.scan_start + i
        points, _ = load_kitti_scan(scan_id)

        # Calcular delta_pose si no es primer frame
        if i == 0:
            delta_pose = None
        else:
            delta_pose = LidarPipelineSuite.compute_delta_pose(
                poses[args.scan_start + i - 1],
                poses[args.scan_start + i]
            )

        result_ego = pipeline_ego.stage3_per_point(points, delta_pose=delta_pose)
        timing_ego_total += result_ego['timing_total_ms']

        if i == args.n_frames - 1:
            result_ego_final = result_ego

        if i % 5 == 0:
            obs_count = np.sum(result_ego['obs_mask'])
            print(f"  Frame {scan_id}: {obs_count} obstacles detectados")

    metrics_ego = compute_detection_metrics(
        gt_mask=gt_mask_ref,
        pred_mask=result_ego_final['obs_mask']
    )

    print()
    print(f"✓ Obstacles detectados (último frame): {np.sum(result_ego_final['obs_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {100*metrics_ego['precision']:.2f}%")
    print(f"    Recall:    {100*metrics_ego['recall']:.2f}%")
    print(f"    F1 Score:  {100*metrics_ego['f1']:.2f}%")
    print(f"    TP: {metrics_ego['tp']}, FP: {metrics_ego['fp']}, FN: {metrics_ego['fn']}")
    print()
    print(f"  Timing total ({args.n_frames} frames): {timing_ego_total:.0f} ms")
    print(f"  Timing promedio por frame: {timing_ego_total/args.n_frames:.0f} ms")
    print()

    # ========================================
    # ANÁLISIS COMPARATIVO
    # ========================================
    print("-" * 80)
    print("ANÁLISIS: EGOMOTION IMPACT")
    print("-" * 80)
    print()

    recall_improvement = 100 * (metrics_ego['recall'] - metrics_no_ego['recall'])
    precision_change = 100 * (metrics_ego['precision'] - metrics_no_ego['precision'])
    f1_improvement = 100 * (metrics_ego['f1'] - metrics_no_ego['f1'])

    print("Cambios CON egomotion vs SIN egomotion:")
    print(f"  Recall:    {recall_improvement:+.2f}%")
    print(f"  Precision: {precision_change:+.2f}%")
    print(f"  F1 Score:  {f1_improvement:+.2f}%")
    print()

    fn_reduction = metrics_no_ego['fn'] - metrics_ego['fn']
    print(f"False Negatives:")
    print(f"  Sin egomotion: {metrics_no_ego['fn']}")
    print(f"  Con egomotion: {metrics_ego['fn']}")
    print(f"  Reduction: {fn_reduction} ({100*fn_reduction/metrics_no_ego['fn']:.1f}%)")
    print()

    timing_overhead = timing_ego_total/args.n_frames - timing_no_ego_total/args.n_frames
    print(f"Overhead de egomotion:")
    print(f"  {timing_overhead:+.0f} ms ({100*timing_overhead/(timing_no_ego_total/args.n_frames):+.1f}%)")
    print()

    # ========================================
    # TABLA COMPARATIVA
    # ========================================
    print("=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print()

    print("┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric              │   Stage 2    │Stage 3 No Ego│Stage 3 W/ Ego│")
    print("│                     │  (Baseline)  │   (KDTree)   │  (KDTree+E)  │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Obstacles Detected  │{np.sum(result_stage2['obs_mask']):>14} │{np.sum(result_no_ego_final['obs_mask']):>14} │{np.sum(result_ego_final['obs_mask']):>14} │")
    print(f"│ Ground Truth Obs    │{np.sum(gt_mask_ref):>14} │{np.sum(gt_mask_ref):>14} │{np.sum(gt_mask_ref):>14} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Precision           │{100*metrics_stage2['precision']:>13.2f}% │{100*metrics_no_ego['precision']:>13.2f}% │{100*metrics_ego['precision']:>13.2f}% │")
    print(f"│ Recall              │{100*metrics_stage2['recall']:>13.2f}% │{100*metrics_no_ego['recall']:>13.2f}% │{100*metrics_ego['recall']:>13.2f}% │")
    print(f"│ F1 Score            │{100*metrics_stage2['f1']:>13.2f}% │{100*metrics_no_ego['f1']:>13.2f}% │{100*metrics_ego['f1']:>13.2f}% │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ True Positives      │{metrics_stage2['tp']:>14} │{metrics_no_ego['tp']:>14} │{metrics_ego['tp']:>14} │")
    print(f"│ False Positives     │{metrics_stage2['fp']:>14} │{metrics_no_ego['fp']:>14} │{metrics_ego['fp']:>14} │")
    print(f"│ False Negatives     │{metrics_stage2['fn']:>14} │{metrics_no_ego['fn']:>14} │{metrics_ego['fn']:>14} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ FN Reduction        │          N/A │            0 │{fn_reduction:>14} │")
    print(f"│ FN Reduction %      │          N/A │          0.0% │{100*fn_reduction/metrics_no_ego['fn']:>13.1f}% │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Timing (ms)         │{result_stage2['timing_total_ms']:>14.0f} │{timing_no_ego_total/args.n_frames:>14.0f} │{timing_ego_total/args.n_frames:>14.0f} │")
    print(f"│ Frames Processed    │{1:>14} │{args.n_frames:>14} │{args.n_frames:>14} │")
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # ========================================
    # CONCLUSIONES
    # ========================================
    print("=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)
    print()

    if metrics_ego['recall'] > metrics_no_ego['recall']:
        print(f"✓ RECALL MEJORÓ: {100*metrics_ego['recall']:.2f}% (>{100*metrics_no_ego['recall']:.2f}%)")
        print(f"  → Egomotion compensation mejora asociación temporal")
        print(f"  → FN reducidos: {fn_reduction} ({100*fn_reduction/metrics_no_ego['fn']:.1f}%)")
    else:
        print(f"⚠ RECALL NO MEJORÓ: {100*metrics_ego['recall']:.2f}% (<{100*metrics_no_ego['recall']:.2f}%)")
        print(f"  → Revisar delta_pose calculation")

    print()

    if f1_improvement > 0:
        print(f"✓ F1 Score MEJORÓ: +{f1_improvement:.2f}%")
    else:
        print(f"⚠ F1 Score NO MEJORÓ: {f1_improvement:.2f}%")

    print()

    if timing_overhead < 10:
        print(f"✓ Overhead ACEPTABLE: {timing_overhead:+.0f} ms (<10ms)")
    else:
        print(f"⚠ Overhead ALTO: {timing_overhead:+.0f} ms")
        print(f"  → Revisar eficiencia de delta_pose transform")

    print()

    print("→ RECOMENDACIÓN: ", end="")
    if metrics_ego['recall'] > metrics_no_ego['recall'] + 0.01:  # >1% mejora
        print("✅ USAR EGOMOTION")
        print("  Recall mejora significativamente con egomotion compensation")
    else:
        print("⚠️ REVISAR IMPLEMENTACIÓN")
        print("  Egomotion NO mejora recall como esperado")

    print()
    print("=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)


if __name__ == '__main__':
    main()
