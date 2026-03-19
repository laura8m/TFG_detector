#!/usr/bin/env python3
"""
Test de Stage 3: Bayesian Temporal Filter
==========================================

Evalúa el filtrado temporal Bayesiano y su capacidad de reducir falsos positivos.

Configuraciones evaluadas:
1. Stage 2 solo (single-frame, sin filtrado temporal)
2. Stage 3 multi-frame simulado (3 frames idénticos para ver acumulación)
3. Stage 3 multi-frame real (diferentes frames secuenciales)

Métricas:
- Obstacle detection: Precision, Recall, F1
- Temporal consistency: estabilidad de detecciones
- False Positive reduction: FPs eliminados por filtro temporal

Basado en:
- ALGORITMO_OPTIMO_V4.md líneas 595-625 (Stage 3A)
- lidar_pipeline_suite.py stage3_complete() method

Autor: TFG LiDAR Geometry
Fecha: Marzo 2026
"""

import numpy as np
import sys
from pathlib import Path

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

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN)
    }


def belief_mask_to_point_mask(belief_mask, range_proj):
    """
    Convierte máscara 2D (H×W) de belief_map a máscara 1D (N) de puntos.

    Args:
        belief_mask: (H, W) máscara booleana en range image
        range_proj: Dict con u, v, valid_mask

    Returns:
        (N,) máscara booleana para puntos originales
    """
    u = range_proj['u']
    v = range_proj['v']
    valid_mask = range_proj['valid_mask']

    point_mask = np.zeros(len(u), dtype=bool)

    # Para cada punto válido, consultar belief_mask
    valid_idx = np.where(valid_mask)[0]
    for idx in valid_idx:
        point_mask[idx] = belief_mask[u[idx], v[idx]]

    return point_mask


# ================================================================================
# TEST: STAGE 3 BAYESIAN FILTER
# ================================================================================

def test_stage3_bayesian_filter(scan_start=0, n_frames=1):
    """
    Test de Stage 3: Bayesian Temporal Filter.

    Args:
        scan_start: ID del primer scan
        n_frames: Número de frames a procesar (1=single-frame, >1=multi-frame)

    Configuraciones:
    1. Stage 2 solo (baseline, sin filtrado temporal)
    2. Stage 3 (Bayesian filter con n_frames)

    Métricas:
    - Precision, Recall, F1
    - False Positive Reduction por filtro temporal
    """
    print("=" * 80)
    print("TEST: STAGE 3 BAYESIAN TEMPORAL FILTER")
    print("=" * 80)
    print(f"Scan range: {scan_start} - {scan_start + n_frames - 1}")
    print(f"Frames to process: {n_frames}")
    print()

    # ========================================
    # CARGAR DATOS
    # ========================================

    # Cargar primer frame solo para info
    points_first, semantic_labels_first = load_kitti_scan(scan_start)
    gt_masks_first = get_ground_truth_masks(semantic_labels_first)

    print(f"✓ Primer frame (scan {scan_start}): {len(points_first)} puntos")
    print(f"  Ground truth obstacles: {gt_masks_first['obstacle'].sum()}")
    print()

    # Cargar ÚLTIMO frame para evaluación (así comparamos Stage 2 vs Stage 3 en mismo frame)
    scan_last = scan_start + n_frames - 1
    points_ref, semantic_labels_ref = load_kitti_scan(scan_last)
    gt_masks_ref = get_ground_truth_masks(semantic_labels_ref)

    print(f"✓ Último frame (scan {scan_last}): {len(points_ref)} puntos")
    print(f"  Ground truth obstacles: {gt_masks_ref['obstacle'].sum()}")
    print()

    # ========================================
    # CONFIG 1: STAGE 2 SOLO (Baseline - sin filtrado temporal)
    # ========================================

    print("-" * 80)
    print(f"CONFIG 1: STAGE 2 SOLO (Baseline - frame {scan_last})")
    print("-" * 80)

    config_stage2 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_temporal_filter=False,  # Sin filtrado temporal
        verbose=False
    )

    pipeline_stage2 = LidarPipelineSuite(config_stage2)
    result_stage2 = pipeline_stage2.stage2_complete(points_ref)

    metrics_stage2 = compute_detection_metrics(
        gt_mask=gt_masks_ref['obstacle'],
        pred_mask=result_stage2['obs_mask']
    )

    print(f"✓ Obstacles detectados (delta_r < -0.3): {np.sum(result_stage2['obs_mask'])}")
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
    # CONFIG 2: STAGE 3 (Bayesian Filter - multi-frame)
    # ========================================

    print("-" * 80)
    print(f"CONFIG 2: STAGE 3 (Bayesian Filter - {n_frames} frames)")
    print("-" * 80)

    config_stage3 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_temporal_filter=True,  # CON filtrado temporal
        gamma=0.7,  # AJUSTADO: más responsive
        prob_threshold_obs=0.35,  # AJUSTADO: más sensible
        verbose=True  # ACTIVADO: para debug
    )

    pipeline_stage3 = LidarPipelineSuite(config_stage3)

    # Procesar múltiples frames acumulando belief
    result_stage3_final = None
    timing_stage3_total = 0.0

    for i in range(n_frames):
        scan_id = scan_start + i
        points, semantic_labels = load_kitti_scan(scan_id)

        # Procesar frame (sin egomotion por ahora, TODO)
        result_stage3 = pipeline_stage3.stage3_complete(points, delta_pose=None)
        timing_stage3_total += result_stage3['timing_total_ms']

        # Guardar resultado del último frame
        if i == n_frames - 1:
            result_stage3_final = result_stage3

        if config_stage3.verbose or (i % 5 == 0):
            obs_count = np.sum(result_stage3['obs_belief_mask'])
            print(f"  Frame {scan_id}: {obs_count} obstacles en belief_map")

    # Convertir belief_mask (H×W) a point_mask (N) para métricas
    obs_belief_point_mask = belief_mask_to_point_mask(
        result_stage3_final['obs_belief_mask'],
        result_stage3_final['range_proj']
    )

    metrics_stage3 = compute_detection_metrics(
        gt_mask=gt_masks_ref['obstacle'],
        pred_mask=obs_belief_point_mask
    )

    print()
    print(f"✓ Obstacles en belief_map (último frame): {np.sum(result_stage3_final['obs_belief_mask'])}")
    print(f"  Obstacles como puntos: {np.sum(obs_belief_point_mask)}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_stage3['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_stage3['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_stage3['f1']*100:.2f}%")
    print(f"    TP: {metrics_stage3['TP']}, FP: {metrics_stage3['FP']}, FN: {metrics_stage3['FN']}")
    print()
    print(f"  Timing total ({n_frames} frames): {timing_stage3_total:.0f} ms")
    print(f"  Timing promedio por frame: {timing_stage3_total/n_frames:.0f} ms")
    print()

    # ========================================
    # ANÁLISIS: FALSE POSITIVE REDUCTION
    # ========================================

    print("-" * 80)
    print("ANÁLISIS: FALSE POSITIVE REDUCTION")
    print("-" * 80)
    print()

    fp_reduction = metrics_stage2['FP'] - metrics_stage3['FP']
    fp_reduction_pct = (fp_reduction / metrics_stage2['FP']) * 100 if metrics_stage2['FP'] > 0 else 0.0

    print(f"False Positives:")
    print(f"  Stage 2 (single-frame): {metrics_stage2['FP']}")
    print(f"  Stage 3 ({n_frames}-frame):   {metrics_stage3['FP']}")
    print(f"  Reduction: {fp_reduction} ({fp_reduction_pct:+.1f}%)")
    print()

    precision_improvement = (metrics_stage3['precision'] - metrics_stage2['precision']) * 100
    f1_improvement = (metrics_stage3['f1'] - metrics_stage2['f1']) * 100

    print(f"Precision improvement: {precision_improvement:+.2f}%")
    print(f"F1 improvement: {f1_improvement:+.2f}%")
    print()

    # ========================================
    # TABLA COMPARATIVA
    # ========================================

    print("=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print()
    print("┌─────────────────────┬──────────────┬──────────────┐")
    print("│ Metric              │   Stage 2    │   Stage 3    │")
    print("│                     │ (Single-frame│ (Multi-frame)│")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Obstacles Detected  │ {np.sum(result_stage2['obs_mask']):>12} │ {np.sum(obs_belief_point_mask):>12} │")
    print(f"│ Ground Truth Obs    │ {gt_masks_ref['obstacle'].sum():>12} │ {gt_masks_ref['obstacle'].sum():>12} │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Precision           │ {metrics_stage2['precision']*100:>11.2f}% │ {metrics_stage3['precision']*100:>11.2f}% │")
    print(f"│ Recall              │ {metrics_stage2['recall']*100:>11.2f}% │ {metrics_stage3['recall']*100:>11.2f}% │")
    print(f"│ F1 Score            │ {metrics_stage2['f1']*100:>11.2f}% │ {metrics_stage3['f1']*100:>11.2f}% │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ True Positives      │ {metrics_stage2['TP']:>12} │ {metrics_stage3['TP']:>12} │")
    print(f"│ False Positives     │ {metrics_stage2['FP']:>12} │ {metrics_stage3['FP']:>12} │")
    print(f"│ False Negatives     │ {metrics_stage2['FN']:>12} │ {metrics_stage3['FN']:>12} │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ FP Reduction        │          N/A │ {fp_reduction:>12} │")
    print(f"│ FP Reduction %      │          N/A │ {fp_reduction_pct:>11.1f}% │")
    print("├─────────────────────┼──────────────┼──────────────┤")
    print(f"│ Timing (ms)         │ {result_stage2['timing_total_ms']:>12.0f} │ {timing_stage3_total/n_frames:>12.0f} │")
    print(f"│ Frames Processed    │            1 │ {n_frames:>12} │")
    print("└─────────────────────┴──────────────┴──────────────┘")
    print()

    # ========================================
    # CONCLUSIONES
    # ========================================

    print("=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)
    print()

    if fp_reduction_pct > 20:
        print(f"✓ Stage 3 ALTAMENTE EFECTIVO: Reduce FPs en {fp_reduction_pct:.1f}%")
    elif fp_reduction_pct > 10:
        print(f"✓ Stage 3 EFECTIVO: Reduce FPs en {fp_reduction_pct:.1f}%")
    elif fp_reduction_pct > 0:
        print(f"~ Stage 3 MODERADO: Reduce FPs en {fp_reduction_pct:.1f}%")
    else:
        print(f"✗ Stage 3 NO EFECTIVO: No reduce FPs ({fp_reduction_pct:.1f}%)")

    print()

    if f1_improvement > 5:
        print(f"✓ F1 Score MEJORA SIGNIFICATIVA: +{f1_improvement:.2f}%")
    elif f1_improvement > 2:
        print(f"✓ F1 Score MEJORA MODERADA: +{f1_improvement:.2f}%")
    elif f1_improvement > 0:
        print(f"~ F1 Score MEJORA LEVE: +{f1_improvement:.2f}%")
    else:
        print(f"✗ F1 Score NO MEJORA: {f1_improvement:+.2f}%")

    print()

    avg_timing_stage3 = timing_stage3_total / n_frames
    overhead_pct = ((avg_timing_stage3 - result_stage2['timing_total_ms']) / result_stage2['timing_total_ms']) * 100

    print(f"Overhead de Stage 3: +{avg_timing_stage3 - result_stage2['timing_total_ms']:.0f} ms ({overhead_pct:.1f}%)")

    if overhead_pct < 20 and fp_reduction_pct > 10:
        print("→ RECOMENDACIÓN: Stage 3 RECOMENDADO (bajo overhead, buena reducción FP)")
    elif fp_reduction_pct > 20:
        print("→ RECOMENDACIÓN: Stage 3 RECOMENDADO (excelente reducción FP)")
    elif overhead_pct < 30:
        print("→ RECOMENDACIÓN: Stage 3 ACEPTABLE (overhead moderado)")
    else:
        print("→ RECOMENDACIÓN: Stage 3 DISCUTIBLE (alto overhead para beneficio limitado)")

    print()
    print("=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Stage 3: Bayesian Temporal Filter")
    parser.add_argument("--scan_start", type=int, default=0, help="ID del primer scan (default: 0)")
    parser.add_argument("--n_frames", type=int, default=1, help="Número de frames a procesar (default: 1)")
    args = parser.parse_args()

    test_stage3_bayesian_filter(scan_start=args.scan_start, n_frames=args.n_frames)
