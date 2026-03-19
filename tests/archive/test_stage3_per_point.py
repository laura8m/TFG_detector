#!/usr/bin/env python3
"""
Test de Stage 3 Per-Point: Bayesian Temporal Filter sin range image.

Evalúa la solución óptima que evita compresión 20:1 usando KDTree.

Comparación:
1. Stage 2 solo (baseline)
2. Stage 3 range image (CLOSEST WINS - problema)
3. Stage 3 per-point (KDTree - solución)

Métricas:
- Obstacle detection: Precision, Recall, F1
- Temporal consistency: con/sin egomotion
- Timing: overhead de KDTree vs warpAffine

Autor: TFG LiDAR Geometry
Fecha: Marzo 2026
"""

import numpy as np
import sys
from pathlib import Path
import argparse

# Agregar paths necesarios
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


# ================================================================================
# UTILIDADES
# ================================================================================

def load_kitti_scan(scan_id=0):
    """Carga scan KITTI (.bin) y labels SemanticKITTI (.label)."""
    velodyne_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne")
    labels_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels")

    scan_file = velodyne_path / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]

    label_file = labels_path / f"{scan_id:06d}.label"
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF

    return points, semantic_labels


def get_ground_truth_masks(semantic_labels):
    """Genera máscaras de ground truth."""
    obstacle_classes = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,  # Persons/cyclists
        50, 51, 52,  # Buildings/fences/walls
        60, 70, 71,  # Trunk, vegetation
        80, 81,  # Poles, traffic signs
        252, 253, 254, 255, 256, 257, 258, 259  # Other movable
    ]
    obstacle_mask = np.isin(semantic_labels, obstacle_classes)
    ground_mask = np.isin(semantic_labels, [40, 44, 48, 49, 72])

    return {
        'obstacle': obstacle_mask,
        'ground': ground_mask
    }


def compute_detection_metrics(gt_mask, pred_mask):
    """Calcula Precision, Recall, F1."""
    TP = np.sum(gt_mask & pred_mask)
    FP = np.sum((~gt_mask) & pred_mask)
    FN = np.sum(gt_mask & (~pred_mask))
    TN = np.sum((~gt_mask) & (~pred_mask))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


# ================================================================================
# MAIN TEST
# ================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=1, help='Número de frames para temporal filtering')
    args = parser.parse_args()

    print("=" * 80)
    print("TEST: STAGE 3 PER-POINT (Bayesian Filter sin range image)")
    print("=" * 80)
    print(f"Scan range: {args.scan_start} - {args.scan_start + args.n_frames - 1}")
    print(f"Frames to process: {args.n_frames}")
    print()

    # ========================================
    # CARGAR DATOS
    # ========================================

    # Cargar ÚLTIMO frame para evaluación
    scan_last = args.scan_start + args.n_frames - 1
    points_ref, semantic_labels_ref = load_kitti_scan(scan_last)
    gt_masks_ref = get_ground_truth_masks(semantic_labels_ref)

    print(f"✓ Último frame (scan {scan_last}): {len(points_ref)} puntos")
    print(f"  Ground truth obstacles: {gt_masks_ref['obstacle'].sum()}")
    print()

    # ========================================
    # CONFIG 1: STAGE 2 SOLO (Baseline)
    # ========================================

    print("-" * 80)
    print(f"CONFIG 1: STAGE 2 SOLO (Baseline - frame {scan_last})")
    print("-" * 80)

    config_stage2 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_temporal_filter=False,
        verbose=False
    )

    pipeline_stage2 = LidarPipelineSuite(config_stage2)
    result_stage2 = pipeline_stage2.stage2_complete(points_ref)

    metrics_stage2 = compute_detection_metrics(
        gt_mask=gt_masks_ref['obstacle'],
        pred_mask=result_stage2['obs_mask']
    )

    print(f"✓ Obstacles detectados: {np.sum(result_stage2['obs_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_stage2['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_stage2['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_stage2['f1']*100:.2f}%")
    print(f"    TP: {metrics_stage2['TP']}, FP: {metrics_stage2['FP']}, FN: {metrics_stage2['FN']}")
    print()
    print(f"  Timing: {result_stage2['timing_total_ms']:.0f} ms")
    print()

    # ========================================
    # CONFIG 2: STAGE 3 PER-POINT (KDTree - NUEVO)
    # ========================================

    print("-" * 80)
    print(f"CONFIG 2: STAGE 3 PER-POINT (KDTree - {args.n_frames} frames)")
    print("-" * 80)

    config_stage3_per_point = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_temporal_filter=True,
        prob_threshold_obs=0.35,
        verbose=True  # Ver debug
    )

    pipeline_per_point = LidarPipelineSuite(config_stage3_per_point)

    # Procesar múltiples frames acumulando belief (sin egomotion por ahora)
    result_per_point_final = None
    timing_per_point_total = 0.0

    for i in range(args.n_frames):
        scan_id = args.scan_start + i
        points, semantic_labels = load_kitti_scan(scan_id)

        # Procesar frame (sin egomotion por ahora, TODO: agregar poses)
        result_per_point = pipeline_per_point.stage3_per_point(points, delta_pose=None)
        timing_per_point_total += result_per_point['timing_total_ms']

        # Guardar resultado del último frame
        if i == args.n_frames - 1:
            result_per_point_final = result_per_point

        if config_stage3_per_point.verbose or (i % 5 == 0):
            obs_count = np.sum(result_per_point['obs_mask'])
            print(f"  Frame {scan_id}: {obs_count} obstacles detectados")

    metrics_per_point = compute_detection_metrics(
        gt_mask=gt_masks_ref['obstacle'],
        pred_mask=result_per_point_final['obs_mask']
    )

    print()
    print(f"✓ Obstacles detectados (último frame): {np.sum(result_per_point_final['obs_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_per_point['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_per_point['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_per_point['f1']*100:.2f}%")
    print(f"    TP: {metrics_per_point['TP']}, FP: {metrics_per_point['FP']}, FN: {metrics_per_point['FN']}")
    print()
    print(f"  Timing total ({args.n_frames} frames): {timing_per_point_total:.0f} ms")
    print(f"  Timing promedio por frame: {timing_per_point_total/args.n_frames:.0f} ms")
    print()

    # ========================================
    # ANÁLISIS COMPARATIVO
    # ========================================

    print("-" * 80)
    print("ANÁLISIS: STAGE 3 PER-POINT vs STAGE 2")
    print("-" * 80)
    print()

    # Cambios en métricas
    recall_change = (metrics_per_point['recall'] - metrics_stage2['recall']) * 100
    precision_change = (metrics_per_point['precision'] - metrics_stage2['precision']) * 100
    f1_change = (metrics_per_point['f1'] - metrics_stage2['f1']) * 100

    print("Cambios respecto a Stage 2:")
    print(f"  Recall:    {recall_change:+.2f}%")
    print(f"  Precision: {precision_change:+.2f}%")
    print(f"  F1 Score:  {f1_change:+.2f}%")
    print()

    # False Positives
    fp_reduction = metrics_stage2['FP'] - metrics_per_point['FP']
    fp_reduction_pct = 100 * fp_reduction / metrics_stage2['FP'] if metrics_stage2['FP'] > 0 else 0.0

    print(f"False Positives:")
    print(f"  Stage 2:          {metrics_stage2['FP']}")
    print(f"  Stage 3 Per-Point: {metrics_per_point['FP']}")
    print(f"  Reduction: {fp_reduction:+d} ({fp_reduction_pct:+.1f}%)")
    print()

    # Timing overhead
    timing_overhead = timing_per_point_total/args.n_frames - result_stage2['timing_total_ms']
    timing_overhead_pct = 100 * timing_overhead / result_stage2['timing_total_ms']

    print(f"Overhead de Stage 3 Per-Point:")
    print(f"  +{timing_overhead:.0f} ms ({timing_overhead_pct:+.1f}%)")
    print()

    # ========================================
    # TABLA COMPARATIVA
    # ========================================

    print("=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print()

    print("┌─────────────────────┬──────────────┬──────────────┐")
    print("│ Metric              │   Stage 2    │Stage 3 PerPt │")
    print("│                     │  (Baseline)  │   (KDTree)   │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Obstacles Detected  │{metrics_stage2['TP']+metrics_stage2['FP']:14d} │{metrics_per_point['TP']+metrics_per_point['FP']:14d} │")
    print(f"│ Ground Truth Obs    │{gt_masks_ref['obstacle'].sum():14d} │{gt_masks_ref['obstacle'].sum():14d} │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Precision           │{metrics_stage2['precision']*100:13.2f}% │{metrics_per_point['precision']*100:13.2f}% │")
    print(f"│ Recall              │{metrics_stage2['recall']*100:13.2f}% │{metrics_per_point['recall']*100:13.2f}% │")
    print(f"│ F1 Score            │{metrics_stage2['f1']*100:13.2f}% │{metrics_per_point['f1']*100:13.2f}% │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ True Positives      │{metrics_stage2['TP']:14d} │{metrics_per_point['TP']:14d} │")
    print(f"│ False Positives     │{metrics_stage2['FP']:14d} │{metrics_per_point['FP']:14d} │")
    print(f"│ False Negatives     │{metrics_stage2['FN']:14d} │{metrics_per_point['FN']:14d} │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ FP Reduction        │          N/A │{fp_reduction:14d} │")
    print(f"│ FP Reduction %      │          N/A │{fp_reduction_pct:13.1f}% │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Timing (ms)         │{result_stage2['timing_total_ms']:14.0f} │{timing_per_point_total/args.n_frames:14.0f} │")
    print(f"│ Frames Processed    │            1 │{args.n_frames:14d} │")
    print("└─────────────────────┴──────────────┴──────────────┘")
    print()

    # ========================================
    # CONCLUSIONES
    # ========================================

    print("=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)
    print()

    # Evaluar recall
    if metrics_per_point['recall'] > 0.90:
        print(f"✓ RECALL EXCELENTE: {metrics_per_point['recall']*100:.2f}% (>90%)")
        print("  → Stage 3 Per-Point mantiene recall de Stage 2")
        print("  → NO sufre compresión 20:1 (sin range image)")
    elif metrics_per_point['recall'] > 0.80:
        print(f"✓ RECALL BUENO: {metrics_per_point['recall']*100:.2f}% (80-90%)")
        print("  → Stage 3 Per-Point reduce ligeramente recall")
    else:
        print(f"✗ RECALL BAJO: {metrics_per_point['recall']*100:.2f}% (<80%)")
        print("  → Algo salió mal, investigar")

    print()

    # Evaluar F1
    if f1_change > 0:
        print(f"✓ F1 Score MEJORA: {f1_change:+.2f}%")
    else:
        print(f"✗ F1 Score NO MEJORA: {f1_change:+.2f}%")

    print()

    # Evaluar overhead
    if timing_overhead_pct < 10:
        print(f"✓ Overhead ACEPTABLE: +{timing_overhead:.0f} ms ({timing_overhead_pct:+.1f}%)")
        print("  → KDTree es eficiente")
    else:
        print(f"⚠ Overhead ALTO: +{timing_overhead:.0f} ms ({timing_overhead_pct:+.1f}%)")
        print("  → Optimizar KDTree query")

    print()

    # Recomendación final
    if metrics_per_point['recall'] > 0.90 and metrics_per_point['f1'] > metrics_stage2['f1']:
        print("→ RECOMENDACIÓN: ✅ USAR STAGE 3 PER-POINT")
        print("  Mantiene recall y mejora F1, overhead moderado")
    elif metrics_per_point['recall'] > 0.90:
        print("→ RECOMENDACIÓN: ✅ CONSIDERAR STAGE 3 PER-POINT")
        print("  Mantiene recall, F1 similar, sin compresión 20:1")
    else:
        print("→ RECOMENDACIÓN: ⚠ INVESTIGAR PROBLEMA")
        print("  Recall no se mantiene como esperado")

    print()
    print("=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)


if __name__ == '__main__':
    main()
