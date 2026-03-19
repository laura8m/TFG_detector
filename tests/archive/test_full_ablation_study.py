#!/usr/bin/env python3
"""
Test de Ablation Study Completo: Baseline → +WallRej → +HCD
===========================================================

Compara tres configuraciones del pipeline Stage 1:
1. Patchwork++ Baseline (sin wall rejection, sin HCD)
2. Patchwork++ + Wall Rejection Hybrid V2.3
3. Patchwork++ + Wall Rejection + HCD (pipeline completo)

Métricas evaluadas contra ground truth SemanticKITTI:
- Wall Rejection: Recall, Precision, F1, Timing
- HCD: Overhead de cómputo, calidad del descriptor
- Comparación detallada por configuración

Basado en:
- test_lidar_pipeline_suite.py líneas 192-283 (test_ablation_study)
- Conversación: "prueba ahora con HC, haz un test que compare patchwork base,
  luego con walls rejection hybrid y luego además con HCD"

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

    Returns:
        Dict con máscaras booleanas:
        - 'wall': clases 50, 51, 52 (building, fence, other-structure)
        - 'ground': clases 40, 44, 48, 49, 72 (road, parking, sidewalk, etc.)
    """
    # Wall classes (según SemanticKITTI)
    wall_mask = np.isin(semantic_labels, [50, 51, 52])

    # Ground classes (según SemanticKITTI)
    ground_mask = np.isin(semantic_labels, [40, 44, 48, 49, 72])

    return {
        'wall': wall_mask,
        'ground': ground_mask
    }


def compute_wall_rejection_metrics(gt_wall_mask, predicted_ground_indices, rejected_wall_indices, n_points):
    """
    Calcula métricas de wall rejection contra ground truth.

    Args:
        gt_wall_mask: (N,) máscara booleana de walls en ground truth
        predicted_ground_indices: (M,) índices clasificados como ground por Patchwork++
        rejected_wall_indices: (K,) índices rechazados por wall rejection
        n_points: Total de puntos en scan

    Returns:
        Dict con recall, precision, F1, TP, FP, FN
    """
    # Convertir a sets para operaciones de conjuntos
    predicted_ground_set = set(predicted_ground_indices)
    rejected_set = set(rejected_wall_indices)
    gt_wall_set = set(np.where(gt_wall_mask)[0])

    # Wall FPs de Patchwork++ (walls etiquetadas como ground)
    wall_fps_patchwork = gt_wall_set & predicted_ground_set

    # True Positives: walls correctamente rechazadas
    TP = len(wall_fps_patchwork & rejected_set)

    # False Positives: puntos NO-wall rechazados incorrectamente
    FP = len(rejected_set - wall_fps_patchwork)

    # False Negatives: walls NO detectadas (quedan en ground final)
    FN = len(wall_fps_patchwork - rejected_set)

    # Métricas
    recall = TP / len(wall_fps_patchwork) if len(wall_fps_patchwork) > 0 else 0.0
    precision = TP / len(rejected_set) if len(rejected_set) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'wall_fps_patchwork': len(wall_fps_patchwork)
    }


# ================================================================================
# TEST: ABLATION STUDY COMPLETO
# ================================================================================

def test_full_ablation_study(scan_id=0):
    """
    Ablation study completo: Baseline → +WallRej → +HCD

    Configuraciones:
    1. Baseline: Patchwork++ solo
    2. +WallRej: Baseline + Hybrid Wall Rejection V2.3
    3. +HCD: +WallRej + Height Coding Descriptor

    Métricas:
    - Wall rejection: Recall, Precision, F1 (contra ground truth)
    - HCD: Overhead de cómputo
    - Timing: ms por stage
    """
    print("=" * 80)
    print("TEST: ABLATION STUDY COMPLETO - Baseline → +WallRej → +HCD")
    print("=" * 80)
    print(f"Scan ID: {scan_id}")
    print()

    # ========================================
    # CARGAR DATOS
    # ========================================

    points, semantic_labels = load_kitti_scan(scan_id)
    gt_masks = get_ground_truth_masks(semantic_labels)

    print(f"✓ Cargado scan {scan_id}: {len(points)} puntos")
    print(f"  Ground truth walls: {gt_masks['wall'].sum()}")
    print(f"  Ground truth ground: {gt_masks['ground'].sum()}")
    print()

    # ========================================
    # CONFIG 1: BASELINE (Patchwork++ solo)
    # ========================================

    print("-" * 80)
    print("CONFIG 1: BASELINE (Patchwork++ solo)")
    print("-" * 80)

    config_baseline = PipelineConfig(
        enable_hybrid_wall_rejection=False,
        enable_hcd=False
    )

    pipeline_baseline = LidarPipelineSuite(config_baseline)
    result_baseline = pipeline_baseline.stage1_complete(points)

    # Métricas baseline
    wall_fps_baseline = len(set(np.where(gt_masks['wall'])[0]) & set(result_baseline['ground_indices']))

    print(f"✓ Ground points detectados: {len(result_baseline['ground_indices'])}")
    print(f"  Wall FPs (Patchwork++ error): {wall_fps_baseline}")
    print(f"  Timing: {result_baseline['timing_ms']:.0f} ms")
    print()

    # ========================================
    # CONFIG 2: +WALL REJECTION HYBRID V2.3
    # ========================================

    print("-" * 80)
    print("CONFIG 2: +WALL REJECTION HYBRID V2.3")
    print("-" * 80)

    config_wallrej = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False
    )

    pipeline_wallrej = LidarPipelineSuite(config_wallrej)
    result_wallrej = pipeline_wallrej.stage1_complete(points)

    # Métricas wall rejection
    metrics_wallrej = compute_wall_rejection_metrics(
        gt_wall_mask=gt_masks['wall'],
        predicted_ground_indices=pipeline_wallrej.patchwork.getGroundIndices(),
        rejected_wall_indices=result_wallrej['rejected_walls'],
        n_points=len(points)
    )

    print(f"✓ Ground points (después de wall rejection): {len(result_wallrej['ground_indices'])}")
    print(f"  Walls rechazadas: {len(result_wallrej['rejected_walls'])}")
    print(f"  Wall FPs Patchwork++: {metrics_wallrej['wall_fps_patchwork']}")
    print()
    print("  Métricas Wall Rejection:")
    print(f"    Recall:    {metrics_wallrej['recall']*100:.2f}%")
    print(f"    Precision: {metrics_wallrej['precision']*100:.2f}%")
    print(f"    F1 Score:  {metrics_wallrej['f1']*100:.2f}%")
    print(f"    TP: {metrics_wallrej['TP']}, FP: {metrics_wallrej['FP']}, FN: {metrics_wallrej['FN']}")
    print()
    print(f"  Timing: {result_wallrej['timing_ms']:.0f} ms")
    print(f"    (+{result_wallrej['timing_ms'] - result_baseline['timing_ms']:.0f} ms overhead)")
    print()

    # ========================================
    # CONFIG 3: +HCD (Pipeline completo)
    # ========================================

    print("-" * 80)
    print("CONFIG 3: +HCD (Pipeline completo)")
    print("-" * 80)

    config_full = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True
    )

    pipeline_full = LidarPipelineSuite(config_full)
    result_full = pipeline_full.stage1_complete(points)

    # Métricas HCD
    hcd_nonzero = np.count_nonzero(result_full['hcd'])
    hcd_mean = np.mean(result_full['hcd'])
    hcd_std = np.std(result_full['hcd'])
    hcd_max = np.max(result_full['hcd'])

    # Métricas wall rejection (debería ser igual a CONFIG 2)
    metrics_full = compute_wall_rejection_metrics(
        gt_wall_mask=gt_masks['wall'],
        predicted_ground_indices=pipeline_full.patchwork.getGroundIndices(),
        rejected_wall_indices=result_full['rejected_walls'],
        n_points=len(points)
    )

    print(f"✓ Ground points (final): {len(result_full['ground_indices'])}")
    print(f"  Walls rechazadas: {len(result_full['rejected_walls'])}")
    print()
    print("  Métricas Wall Rejection:")
    print(f"    Recall:    {metrics_full['recall']*100:.2f}%")
    print(f"    Precision: {metrics_full['precision']*100:.2f}%")
    print(f"    F1 Score:  {metrics_full['f1']*100:.2f}%")
    print()
    print("  HCD Descriptor:")
    print(f"    Puntos con HCD > 0: {hcd_nonzero} ({hcd_nonzero/len(result_full['hcd'])*100:.1f}%)")
    print(f"    Mean: {hcd_mean:.4f}, Std: {hcd_std:.4f}, Max: {hcd_max:.4f}")
    print()
    print(f"  Timing: {result_full['timing_ms']:.0f} ms")
    print(f"    (+{result_full['timing_ms'] - result_wallrej['timing_ms']:.0f} ms HCD overhead)")
    print()

    # ========================================
    # TABLA COMPARATIVA
    # ========================================

    print("=" * 80)
    print("TABLA COMPARATIVA")
    print("=" * 80)
    print()
    print("┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Metric              │   Baseline   │   +WallRej   │     +HCD     │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Ground Points       │ {len(result_baseline['ground_indices']):>12} │ {len(result_wallrej['ground_indices']):>12} │ {len(result_full['ground_indices']):>12} │")
    print(f"│ Wall FPs (Patchwork)│ {wall_fps_baseline:>12} │ {metrics_wallrej['wall_fps_patchwork']:>12} │ {metrics_full['wall_fps_patchwork']:>12} │")
    print(f"│ Walls Rejected      │ {0:>12} │ {len(result_wallrej['rejected_walls']):>12} │ {len(result_full['rejected_walls']):>12} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ WallRej Recall      │          N/A │ {metrics_wallrej['recall']*100:>11.2f}% │ {metrics_full['recall']*100:>11.2f}% │")
    print(f"│ WallRej Precision   │          N/A │ {metrics_wallrej['precision']*100:>11.2f}% │ {metrics_full['precision']*100:>11.2f}% │")
    print(f"│ WallRej F1          │          N/A │ {metrics_wallrej['f1']*100:>11.2f}% │ {metrics_full['f1']*100:>11.2f}% │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ HCD Mean            │          N/A │          N/A │ {hcd_mean:>12.4f} │")
    print(f"│ HCD Nonzero %       │          N/A │          N/A │ {hcd_nonzero/len(result_full['hcd'])*100:>11.1f}% │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Timing (ms)         │ {result_baseline['timing_ms']:>12.0f} │ {result_wallrej['timing_ms']:>12.0f} │ {result_full['timing_ms']:>12.0f} │")
    print(f"│ Overhead vs Base    │ {0:>12} │ {result_wallrej['timing_ms'] - result_baseline['timing_ms']:>11.0f}ms │ {result_full['timing_ms'] - result_baseline['timing_ms']:>11.0f}ms │")
    print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
    print()

    # ========================================
    # ANÁLISIS Y CONCLUSIONES
    # ========================================

    print("=" * 80)
    print("ANÁLISIS Y CONCLUSIONES")
    print("=" * 80)
    print()

    # Reducción de Wall FPs
    wall_fp_reduction = (wall_fps_baseline - metrics_full['FN']) / wall_fps_baseline * 100 if wall_fps_baseline > 0 else 0.0

    print("1. WALL REJECTION HYBRID V2.3:")
    print(f"   ✓ Reduce wall FPs de {wall_fps_baseline} → {metrics_full['FN']} ({wall_fp_reduction:.1f}% reducción)")
    print(f"   ✓ Recall {metrics_wallrej['recall']*100:.1f}% (detecta {metrics_wallrej['TP']}/{metrics_wallrej['wall_fps_patchwork']} walls)")
    print(f"   ✓ Precision {metrics_wallrej['precision']*100:.1f}% ({metrics_wallrej['FP']} FPs)")
    print(f"   ✓ Overhead: {result_wallrej['timing_ms'] - result_baseline['timing_ms']:.0f} ms")

    if metrics_wallrej['recall'] > 0.25 and metrics_wallrej['precision'] > 0.15:
        print("   → CONCLUSIÓN: Wall rejection EFECTIVO (recall/precision aceptables)")
    elif metrics_wallrej['recall'] > 0.15:
        print("   → CONCLUSIÓN: Wall rejection MODERADO (recall decente, precision baja)")
    else:
        print("   → CONCLUSIÓN: Wall rejection LIMITADO (baja efectividad)")

    print()
    print("2. HCD (HEIGHT CODING DESCRIPTOR):")
    print(f"   ✓ Genera descriptor para {hcd_nonzero} puntos ({hcd_nonzero/len(result_full['hcd'])*100:.1f}%)")
    print(f"   ✓ Overhead: {result_full['timing_ms'] - result_wallrej['timing_ms']:.0f} ms")

    hcd_overhead_pct = (result_full['timing_ms'] - result_wallrej['timing_ms']) / result_wallrej['timing_ms'] * 100
    if hcd_overhead_pct < 10:
        print(f"   → CONCLUSIÓN: HCD overhead BAJO ({hcd_overhead_pct:.1f}% del tiempo total)")
    elif hcd_overhead_pct < 30:
        print(f"   → CONCLUSIÓN: HCD overhead MODERADO ({hcd_overhead_pct:.1f}% del tiempo total)")
    else:
        print(f"   → CONCLUSIÓN: HCD overhead ALTO ({hcd_overhead_pct:.1f}% del tiempo total)")

    print()
    print("3. PIPELINE COMPLETO (+WallRej +HCD):")
    total_overhead_pct = (result_full['timing_ms'] - result_baseline['timing_ms']) / result_baseline['timing_ms'] * 100
    print(f"   ✓ Overhead total vs baseline: {result_full['timing_ms'] - result_baseline['timing_ms']:.0f} ms ({total_overhead_pct:.0f}% incremento)")
    print(f"   ✓ Wall FP reduction: {wall_fp_reduction:.1f}%")
    print(f"   ✓ Ground points limpios: {len(result_full['ground_indices'])}")

    if wall_fp_reduction > 20 and total_overhead_pct < 50:
        print("   → CONCLUSIÓN: Pipeline completo RECOMENDADO (buena relación accuracy/speed)")
    elif wall_fp_reduction > 15:
        print("   → CONCLUSIÓN: Pipeline completo ACEPTABLE (mejora accuracy con overhead moderado)")
    else:
        print("   → CONCLUSIÓN: Pipeline completo DISCUTIBLE (overhead alto para beneficio limitado)")

    print()
    print("=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Study: Baseline → +WallRej → +HCD")
    parser.add_argument("--scan", type=int, default=0, help="Scan ID (default: 0)")
    args = parser.parse_args()

    test_full_ablation_study(scan_id=args.scan)
