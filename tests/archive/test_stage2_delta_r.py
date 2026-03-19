#!/usr/bin/env python3
"""
Test de Stage 2: Delta-r Anomaly Detection con HCD Fusion
==========================================================

Compara tres configuraciones:
1. Stage 1 solo (baseline para ver ground truth)
2. Stage 2 con Delta-r solo (sin HCD)
3. Stage 2 con Delta-r + HCD Fusion

Métricas evaluadas contra ground truth SemanticKITTI:
- Obstacle Detection: Precision, Recall, F1
- Void Detection: Precision, Recall, F1
- Timing: Overhead de Stage 2

Basado en:
- ALGORITMO_OPTIMO_V4.md líneas 503-573 (Stage 2B)
- lidar_pipeline_suite.py compute_delta_r() method

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
# UTILIDADES: CARGA DE GROUND TRUTH
# ================================================================================

def load_kitti_scan(scan_id=0):
    """Carga scan KITTI (.bin) y labels SemanticKITTI (.label)."""
    # Rutas de datos (sequence 04)
    velodyne_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne")
    labels_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels")

    # Velodyne scan
    scan_file = velodyne_path / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # Solo xyz (sin intensidad)

    # SemanticKITTI labels
    label_file = labels_path / f"{scan_id:06d}.label"
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF  # Lower 16 bits

    return points, semantic_labels


def get_ground_truth_masks(semantic_labels):
    """
    Genera máscaras de ground truth según SemanticKITTI.

    SemanticKITTI obstacle classes:
    - Vehicle: 10, 11, 13, 15, 16, 18, 20
    - Person: 30, 31, 32
    - Cyclist: 31, 32
    - Other movable: 252, 253, 254, 255, 256, 257, 258, 259

    Static obstacles:
    - Building/fence/wall: 50, 51, 52
    - Pole/traffic-sign: 80, 81
    - Vegetation: 70, 71
    - Trunk: 60

    Ground:
    - Road/parking/sidewalk: 40, 44, 48, 49, 72

    Returns:
        Dict con máscaras booleanas:
        - 'obstacle': todos los obstáculos (incluye walls, vehicles, etc.)
        - 'ground': clases de ground
        - 'wall': solo estructuras verticales
    """
    # Obstacle classes (amplio: cualquier cosa no-ground)
    obstacle_classes = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,  # Persons/cyclists
        50, 51, 52,  # Buildings/fences/walls
        60, 70, 71,  # Trunk, vegetation
        80, 81,  # Poles, traffic signs
        252, 253, 254, 255, 256, 257, 258, 259  # Other movable
    ]
    obstacle_mask = np.isin(semantic_labels, obstacle_classes)

    # Ground classes
    ground_mask = np.isin(semantic_labels, [40, 44, 48, 49, 72])

    # Wall classes (subset de obstacles)
    wall_mask = np.isin(semantic_labels, [50, 51, 52])

    return {
        'obstacle': obstacle_mask,
        'ground': ground_mask,
        'wall': wall_mask
    }


def compute_detection_metrics(gt_mask, pred_mask):
    """
    Calcula Precision, Recall, F1 para detección binaria.

    Args:
        gt_mask: (N,) ground truth (True = positivo)
        pred_mask: (N,) predicción (True = detectado)

    Returns:
        Dict con precision, recall, f1, TP, FP, FN
    """
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


# ================================================================================
# TEST: STAGE 2 ABLATION
# ================================================================================

def test_stage2_ablation(scan_id=0):
    """
    Ablation study: Stage 1 → Stage 2 (Delta-r) → Stage 2B (Delta-r+HCD)

    Configuraciones:
    1. Baseline: Solo Stage 1 (sin delta-r)
    2. Stage 2A: Delta-r sin HCD
    3. Stage 2B: Delta-r + HCD Fusion

    Métricas:
    - Obstacle detection: Precision, Recall, F1
    - Timing por stage
    """
    print("=" * 80)
    print("TEST: STAGE 2 ABLATION - Delta-r Solo vs Delta-r+HCD Fusion")
    print("=" * 80)
    print(f"Scan ID: {scan_id}")
    print()

    # ========================================
    # CARGAR DATOS
    # ========================================

    points, semantic_labels = load_kitti_scan(scan_id)
    gt_masks = get_ground_truth_masks(semantic_labels)

    print(f"✓ Cargado scan {scan_id}: {len(points)} puntos")
    print(f"  Ground truth obstacles: {gt_masks['obstacle'].sum()}")
    print(f"  Ground truth ground: {gt_masks['ground'].sum()}")
    print()

    # ========================================
    # CONFIG 1: BASELINE (Stage 1 solo, sin delta-r)
    # ========================================

    print("-" * 80)
    print("CONFIG 1: BASELINE (Stage 1 solo - sin delta-r)")
    print("-" * 80)

    config_baseline = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,  # No HCD para baseline
        verbose=False
    )

    pipeline_baseline = LidarPipelineSuite(config_baseline)
    result_baseline = pipeline_baseline.stage1_complete(points)

    # Baseline: todos los non-ground son "obstáculos detectados"
    baseline_obs_mask = np.ones(len(points), dtype=bool)
    baseline_obs_mask[result_baseline['ground_indices']] = False

    metrics_baseline = compute_detection_metrics(
        gt_mask=gt_masks['obstacle'],
        pred_mask=baseline_obs_mask
    )

    print(f"✓ Ground points: {len(result_baseline['ground_indices'])}")
    print(f"  Non-ground (obstáculos predichos): {np.sum(baseline_obs_mask)}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_baseline['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_baseline['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_baseline['f1']*100:.2f}%")
    print(f"    TP: {metrics_baseline['TP']}, FP: {metrics_baseline['FP']}, FN: {metrics_baseline['FN']}")
    print()
    print(f"  Timing: {result_baseline['timing_ms']:.0f} ms")
    print()

    # ========================================
    # CONFIG 2: STAGE 2A (Delta-r sin HCD)
    # ========================================

    print("-" * 80)
    print("CONFIG 2: STAGE 2A (Delta-r sin HCD)")
    print("-" * 80)

    config_stage2a = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,  # Sin HCD
        verbose=False
    )

    pipeline_stage2a = LidarPipelineSuite(config_stage2a)
    result_stage2a = pipeline_stage2a.stage2_complete(points)

    # Delta-r: obs_mask indica obstáculos detectados
    metrics_stage2a = compute_detection_metrics(
        gt_mask=gt_masks['obstacle'],
        pred_mask=result_stage2a['obs_mask']
    )

    print(f"✓ Obstacles detectados (delta_r < -0.3): {np.sum(result_stage2a['obs_mask'])}")
    print(f"  Voids detectados (delta_r > +0.5): {np.sum(result_stage2a['void_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_stage2a['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_stage2a['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_stage2a['f1']*100:.2f}%")
    print(f"    TP: {metrics_stage2a['TP']}, FP: {metrics_stage2a['FP']}, FN: {metrics_stage2a['FN']}")
    print()
    print(f"  Timing:")
    print(f"    Stage 1: {result_stage2a['timing_ms']:.0f} ms")
    print(f"    Stage 2: {result_stage2a['timing_ms']:.0f} ms")
    print(f"    Total: {result_stage2a['timing_total_ms']:.0f} ms")
    print()

    # ========================================
    # CONFIG 3: STAGE 2B (Delta-r + HCD Fusion)
    # ========================================

    print("-" * 80)
    print("CONFIG 3: STAGE 2B (Delta-r + HCD Fusion)")
    print("-" * 80)

    config_stage2b = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,  # CON HCD
        verbose=False
    )

    pipeline_stage2b = LidarPipelineSuite(config_stage2b)
    result_stage2b = pipeline_stage2b.stage2_complete(points)

    # Delta-r + HCD: usa likelihood mejorada
    # Para comparar, seguimos usando obs_mask (threshold delta_r < -0.3)
    # Pero HCD ha modulado la likelihood interna
    metrics_stage2b = compute_detection_metrics(
        gt_mask=gt_masks['obstacle'],
        pred_mask=result_stage2b['obs_mask']
    )

    # Analizar impacto de HCD en likelihood
    likelihood_delta = result_stage2b['likelihood'] - result_stage2a['likelihood']
    n_boosted = np.sum(likelihood_delta > 0.1)  # Likelihood aumentada significativamente
    n_suppressed = np.sum(likelihood_delta < -0.1)  # Likelihood reducida

    print(f"✓ Obstacles detectados: {np.sum(result_stage2b['obs_mask'])}")
    print(f"  Voids detectados: {np.sum(result_stage2b['void_mask'])}")
    print()
    print("  Métricas Obstacle Detection:")
    print(f"    Precision: {metrics_stage2b['precision']*100:.2f}%")
    print(f"    Recall:    {metrics_stage2b['recall']*100:.2f}%")
    print(f"    F1 Score:  {metrics_stage2b['f1']*100:.2f}%")
    print(f"    TP: {metrics_stage2b['TP']}, FP: {metrics_stage2b['FP']}, FN: {metrics_stage2b['FN']}")
    print()
    print("  HCD Fusion Impact:")
    print(f"    Puntos con likelihood aumentada: {n_boosted}")
    print(f"    Puntos con likelihood reducida: {n_suppressed}")
    print(f"    Mean likelihood delta: {np.mean(likelihood_delta):.3f}")
    print()
    print(f"  Timing:")
    print(f"    Stage 1: {result_stage2b['timing_ms']:.0f} ms")
    print(f"    Stage 2: {result_stage2b['timing_ms']:.0f} ms")
    print(f"    Total: {result_stage2b['timing_total_ms']:.0f} ms")
    print()

    # ========================================
    # TABLA COMPARATIVA
    # ========================================

    print("=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print()
    print("┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric              │   Baseline   │  Stage 2A    │  Stage 2B    │")
    print("│                     │  (Stage 1)   │  (Delta-r)   │ (Delta-r+HCD)│")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Obstacles Detected  │ {np.sum(baseline_obs_mask):>12} │ {np.sum(result_stage2a['obs_mask']):>12} │ {np.sum(result_stage2b['obs_mask']):>12} │")
    print(f"│ Ground Truth Obs    │ {gt_masks['obstacle'].sum():>12} │ {gt_masks['obstacle'].sum():>12} │ {gt_masks['obstacle'].sum():>12} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Precision           │ {metrics_baseline['precision']*100:>11.2f}% │ {metrics_stage2a['precision']*100:>11.2f}% │ {metrics_stage2b['precision']*100:>11.2f}% │")
    print(f"│ Recall              │ {metrics_baseline['recall']*100:>11.2f}% │ {metrics_stage2a['recall']*100:>11.2f}% │ {metrics_stage2b['recall']*100:>11.2f}% │")
    print(f"│ F1 Score            │ {metrics_baseline['f1']*100:>11.2f}% │ {metrics_stage2a['f1']*100:>11.2f}% │ {metrics_stage2b['f1']*100:>11.2f}% │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ True Positives      │ {metrics_baseline['TP']:>12} │ {metrics_stage2a['TP']:>12} │ {metrics_stage2b['TP']:>12} │")
    print(f"│ False Positives     │ {metrics_baseline['FP']:>12} │ {metrics_stage2a['FP']:>12} │ {metrics_stage2b['FP']:>12} │")
    print(f"│ False Negatives     │ {metrics_baseline['FN']:>12} │ {metrics_stage2a['FN']:>12} │ {metrics_stage2b['FN']:>12} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ HCD Boosted         │          N/A │          N/A │ {n_boosted:>12} │")
    print(f"│ HCD Suppressed      │          N/A │          N/A │ {n_suppressed:>12} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Timing (ms)         │ {result_baseline['timing_ms']:>12.0f} │ {result_stage2a['timing_total_ms']:>12.0f} │ {result_stage2b['timing_total_ms']:>12.0f} │")
    print(f"│ Overhead vs Base    │ {0:>12} │ {result_stage2a['timing_total_ms'] - result_baseline['timing_ms']:>11.0f}ms │ {result_stage2b['timing_total_ms'] - result_baseline['timing_ms']:>11.0f}ms │")
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # ========================================
    # ANÁLISIS Y CONCLUSIONES
    # ========================================

    print("=" * 80)
    print("ANÁLISIS Y CONCLUSIONES")
    print("=" * 80)
    print()

    # Mejora de Stage 2A vs Baseline
    prec_improvement_2a = (metrics_stage2a['precision'] - metrics_baseline['precision']) * 100
    f1_improvement_2a = (metrics_stage2a['f1'] - metrics_baseline['f1']) * 100

    print("1. STAGE 2A (Delta-r sin HCD):")
    print(f"   Precision: {metrics_stage2a['precision']*100:.1f}% ({prec_improvement_2a:+.1f}% vs baseline)")
    print(f"   F1 Score: {metrics_stage2a['f1']*100:.1f}% ({f1_improvement_2a:+.1f}% vs baseline)")
    print(f"   Overhead: {result_stage2a['timing_total_ms'] - result_baseline['timing_ms']:.0f} ms")

    if f1_improvement_2a > 5:
        print("   → CONCLUSIÓN: Delta-r EFECTIVO (mejora significativa F1)")
    elif f1_improvement_2a > 0:
        print("   → CONCLUSIÓN: Delta-r MODERADO (mejora leve F1)")
    else:
        print("   → CONCLUSIÓN: Delta-r NO EFECTIVO (F1 no mejora)")

    print()

    # Mejora de Stage 2B vs Stage 2A
    prec_improvement_2b = (metrics_stage2b['precision'] - metrics_stage2a['precision']) * 100
    f1_improvement_2b = (metrics_stage2b['f1'] - metrics_stage2a['f1']) * 100

    print("2. STAGE 2B (Delta-r + HCD Fusion):")
    print(f"   Precision: {metrics_stage2b['precision']*100:.1f}% ({prec_improvement_2b:+.1f}% vs Stage 2A)")
    print(f"   F1 Score: {metrics_stage2b['f1']*100:.1f}% ({f1_improvement_2b:+.1f}% vs Stage 2A)")
    print(f"   Overhead: {result_stage2b['timing_total_ms'] - result_stage2a['timing_total_ms']:.0f} ms")

    if prec_improvement_2b > 2:
        print(f"   → CONCLUSIÓN: HCD Fusion EFECTIVO (+{prec_improvement_2b:.1f}% precision)")
    elif prec_improvement_2b > 0:
        print(f"   → CONCLUSIÓN: HCD Fusion MODERADO (+{prec_improvement_2b:.1f}% precision)")
    else:
        print("   → CONCLUSIÓN: HCD Fusion SIN BENEFICIO (precision no mejora)")

    print()

    # Mejora total
    total_prec_improvement = (metrics_stage2b['precision'] - metrics_baseline['precision']) * 100
    total_f1_improvement = (metrics_stage2b['f1'] - metrics_baseline['f1']) * 100

    print("3. MEJORA TOTAL (Stage 2B vs Baseline):")
    print(f"   Precision: +{total_prec_improvement:.1f}%")
    print(f"   F1 Score: +{total_f1_improvement:.1f}%")
    print(f"   Overhead total: {result_stage2b['timing_total_ms'] - result_baseline['timing_ms']:.0f} ms")

    if total_f1_improvement > 10:
        print("   → CONCLUSIÓN: Stage 2 ALTAMENTE RECOMENDADO")
    elif total_f1_improvement > 5:
        print("   → CONCLUSIÓN: Stage 2 RECOMENDADO")
    elif total_f1_improvement > 0:
        print("   → CONCLUSIÓN: Stage 2 ACEPTABLE")
    else:
        print("   → CONCLUSIÓN: Stage 2 NO JUSTIFICADO")

    print()
    print("=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Stage 2: Delta-r + HCD Fusion")
    parser.add_argument("--scan", type=int, default=0, help="Scan ID (default: 0)")
    args = parser.parse_args()

    test_stage2_ablation(scan_id=args.scan)
