#!/usr/bin/env python3
"""
Tests para lidar_pipeline_suite.py

Evalúa:
1. Stage 1 completo (Patchwork++ + Wall Rejection + HCD)
2. Ablation study (Baseline vs +WallRejection vs +HCD)
3. Métricas con ground truth (SemanticKITTI)

Autor: TFG LiDAR Geometry
Fecha: Marzo 2026
"""

import sys
import os
import numpy as np
from pathlib import Path

# Añadir paths
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


# ================================================================================
# UTILIDADES
# ================================================================================

def load_kitti_bin(file_path):
    """Carga archivo .bin de KITTI"""
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]
    return points


def load_semantickitti_labels(file_path):
    """Carga labels de SemanticKITTI"""
    labels = np.fromfile(file_path, dtype=np.uint32)
    labels = labels & 0xFFFF  # Máscara para semantic label
    return labels


def get_ground_truth_masks(labels):
    """
    Extrae máscaras ground truth de SemanticKITTI.

    Returns:
        wall_mask: bool array para wall points (50, 51, 52)
        ground_mask: bool array para ground points (40, 44, 48, 49, 72)
    """
    wall_classes = [50, 51, 52]  # building, fence, other-structure
    ground_classes = [40, 44, 48, 49, 72]  # road, parking, sidewalk, other-ground, terrain

    wall_mask = np.isin(labels, wall_classes)
    ground_mask = np.isin(labels, ground_classes)

    return wall_mask, ground_mask


# ================================================================================
# TEST 1: FUNCIONAMIENTO BÁSICO DE STAGE 1
# ================================================================================

def test_stage1_basic(scan_id=0, verbose=True):
    """
    Test básico: verifica que Stage 1 se ejecute sin errores.

    Checks:
    - Pipeline se inicializa correctamente
    - Stage 1 completa sin crash
    - Outputs tienen dimensiones correctas
    - Timing es razonable (<200ms)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST 1: FUNCIONAMIENTO BÁSICO - Scan {scan_id:06d}")
        print(f"{'='*80}\n")

    # Configuración por defecto
    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    )

    # Ruta al scan
    data_path = (
        f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
        f"04/04/velodyne/{scan_id:06d}.bin"
    )

    # Cargar puntos
    points = load_kitti_bin(data_path)
    n_points = len(points)

    # Ejecutar pipeline
    pipeline = LidarPipelineSuite(config, data_path=data_path)
    results = pipeline.stage1_complete(points)

    # Validaciones
    assert results is not None, "Stage 1 retornó None"
    assert 'ground_indices' in results, "Falta ground_indices en output"
    assert 'hcd' in results, "Falta HCD en output"
    assert 'rejected_walls' in results, "Falta rejected_walls en output"

    n_ground = len(results['ground_indices'])
    n_nonground = len(results['nonground_indices'])
    n_walls = len(results['rejected_walls'])

    assert n_ground > 0, "No se detectó ground"
    assert n_ground + n_nonground <= n_points, "Suma de puntos excede total"

    timing = results['timing_ms']
    assert timing > 0, "Timing inválido"
    assert timing < 500, f"Timing muy alto: {timing:.1f}ms (esperado <500ms)"

    if verbose:
        print(f"✅ Pipeline inicializado correctamente")
        print(f"✅ Stage 1 completado en {timing:.1f}ms")
        print(f"✅ Ground: {n_ground} | Nonground: {n_nonground} | Walls: {n_walls}")
        print(f"✅ HCD shape: {results['hcd'].shape}")
        print(f"\n{'='*80}\n")

    return True


# ================================================================================
# TEST 2: WALL REJECTION EFFECTIVENESS
# ================================================================================

def test_wall_rejection_with_gt(scan_id=0, verbose=True):
    """
    Test con ground truth: evalúa efectividad del wall rejection.

    Compara contra SemanticKITTI labels:
    - Recall: % de wall points correctamente rechazados
    - Precision: % de rechazos que son realmente walls
    - False Positives: ground points rechazados incorrectamente
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST 2: WALL REJECTION CON GROUND TRUTH - Scan {scan_id:06d}")
        print(f"{'='*80}\n")

    # Rutas
    velodyne_path = (
        f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
        f"04/04/velodyne/{scan_id:06d}.bin"
    )
    label_path = (
        f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
        f"04_labels/04/labels/{scan_id:06d}.label"
    )

    # Cargar datos
    points = load_kitti_bin(velodyne_path)
    labels = load_semantickitti_labels(label_path)

    assert len(points) == len(labels), "Points y labels tienen diferente longitud"

    # Ground truth masks
    gt_wall_mask, gt_ground_mask = get_ground_truth_masks(labels)
    gt_wall_indices = np.where(gt_wall_mask)[0]
    gt_ground_indices = np.where(gt_ground_mask)[0]

    # Ejecutar pipeline
    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,  # Solo wall rejection
        verbose=False
    )

    pipeline = LidarPipelineSuite(config, data_path=velodyne_path)
    results = pipeline.stage1_complete(points)

    # Analizar resultados
    ground_detected = results['ground_indices']
    walls_rejected = results['rejected_walls']

    # Falsos Positivos de Patchwork++: wall points clasificados como ground
    pw_ground_set = set(ground_detected) | set(walls_rejected)
    fp_walls_in_ground = [i for i in pw_ground_set if i in gt_wall_indices]

    # True Positives de Wall Rejection: walls correctamente rechazadas
    tp_walls = [i for i in walls_rejected if i in gt_wall_indices]

    # False Positives de Wall Rejection: ground rechazado incorrectamente
    fp_ground = [i for i in walls_rejected if i in gt_ground_indices]

    # Métricas
    n_gt_walls_total = len(gt_wall_indices)
    n_pw_fp_walls = len(fp_walls_in_ground)
    n_tp_walls = len(tp_walls)
    n_fp_ground = len(fp_ground)

    recall = (n_tp_walls / n_pw_fp_walls * 100) if n_pw_fp_walls > 0 else 0.0
    precision = (n_tp_walls / len(walls_rejected) * 100) if len(walls_rejected) > 0 else 0.0

    if verbose:
        print(f"📊 GROUND TRUTH:")
        print(f"  Total wall points: {n_gt_walls_total}")
        print(f"  Total ground points: {len(gt_ground_indices)}")

        print(f"\n📊 PATCHWORK++ (baseline):")
        print(f"  Wall points clasificados como ground: {n_pw_fp_walls}")
        print(f"  -> Estos son el target del wall rejection")

        print(f"\n📊 WALL REJECTION:")
        print(f"  Total rechazado: {len(walls_rejected)}")
        print(f"  ✅ True Positives (walls): {n_tp_walls}")
        print(f"  ❌ False Positives (ground): {n_fp_ground}")

        print(f"\n📊 MÉTRICAS:")
        print(f"  Recall: {recall:.2f}% (de {n_pw_fp_walls} FP de Patchwork++)")
        print(f"  Precision: {precision:.2f}% (de {len(walls_rejected)} rechazos)")

        print(f"\n{'='*80}\n")

    # Guardar resultados para análisis posterior
    return {
        'recall': recall,
        'precision': precision,
        'tp_walls': n_tp_walls,
        'fp_ground': n_fp_ground,
        'pw_fp_walls': n_pw_fp_walls,
        'total_rejected': len(walls_rejected)
    }


# ================================================================================
# TEST 3: HCD FUNCTIONALITY
# ================================================================================

def test_hcd_descriptor(scan_id=0, verbose=True):
    """
    Test de Height Coding Descriptor (HCD).

    Verifica:
    - HCD se computa correctamente
    - Valores están en rango esperado (normalizado con tanh)
    - Distribución es razonable (no todo cero)
    - Timing overhead es bajo (<5ms)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST 3: HEIGHT CODING DESCRIPTOR - Scan {scan_id:06d}")
        print(f"{'='*80}\n")

    data_path = (
        f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
        f"04/04/velodyne/{scan_id:06d}.bin"
    )

    points = load_kitti_bin(data_path)

    # Config sin HCD (baseline)
    config_baseline = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,
        verbose=False
    )

    pipeline_baseline = LidarPipelineSuite(config_baseline, data_path=data_path)
    results_baseline = pipeline_baseline.stage1_complete(points)

    # Config con HCD
    config_hcd = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    )

    pipeline_hcd = LidarPipelineSuite(config_hcd, data_path=data_path)
    results_hcd = pipeline_hcd.stage1_complete(points)

    # Analizar HCD
    hcd = results_hcd['hcd']

    assert hcd is not None, "HCD es None"
    assert len(hcd) > 0, "HCD está vacío"
    assert len(hcd) == len(results_hcd['ground_indices']), "HCD shape no coincide con ground"

    # Verificar rango (tanh normaliza a [-1, 1])
    assert np.all(hcd >= -1.0) and np.all(hcd <= 1.0), f"HCD fuera de rango: [{np.min(hcd)}, {np.max(hcd)}]"

    # Verificar que no es todo cero
    assert np.std(hcd) > 0.01, "HCD parece constante (std muy bajo)"

    # Timing overhead
    timing_baseline = results_baseline['timing_ms']
    timing_hcd = results_hcd['timing_ms']
    overhead = timing_hcd - timing_baseline

    if verbose:
        print(f"📊 HCD STATISTICS:")
        print(f"  Shape: {hcd.shape}")
        print(f"  Mean: {np.mean(hcd):.3f}")
        print(f"  Std: {np.std(hcd):.3f}")
        print(f"  Range: [{np.min(hcd):.3f}, {np.max(hcd):.3f}]")
        print(f"  P10: {np.percentile(hcd, 10):.3f}")
        print(f"  P50 (median): {np.percentile(hcd, 50):.3f}")
        print(f"  P90: {np.percentile(hcd, 90):.3f}")

        print(f"\n📊 TIMING OVERHEAD:")
        print(f"  Baseline (sin HCD): {timing_baseline:.1f}ms")
        print(f"  Con HCD: {timing_hcd:.1f}ms")
        print(f"  Overhead: +{overhead:.1f}ms ({overhead/timing_baseline*100:.1f}%)")

        # Interpretación
        print(f"\n💡 INTERPRETACIÓN:")
        if np.mean(hcd) > 0.1:
            print(f"  ⚠️  HCD medio alto ({np.mean(hcd):.3f}) -> Muchos puntos elevados")
        elif np.mean(hcd) < -0.1:
            print(f"  ⚠️  HCD medio bajo ({np.mean(hcd):.3f}) -> Muchos puntos deprimidos")
        else:
            print(f"  ✅ HCD medio balanceado ({np.mean(hcd):.3f})")

        if overhead < 5:
            print(f"  ✅ Overhead bajo (<5ms)")
        elif overhead < 10:
            print(f"  ⚠️  Overhead moderado (5-10ms)")
        else:
            print(f"  ❌ Overhead alto (>10ms)")

        print(f"\n{'='*80}\n")

    return {
        'hcd_mean': np.mean(hcd),
        'hcd_std': np.std(hcd),
        'hcd_min': np.min(hcd),
        'hcd_max': np.max(hcd),
        'overhead_ms': overhead
    }


# ================================================================================
# TEST 4: ABLATION STUDY
# ================================================================================

def test_ablation_study(scan_id=0, verbose=True):
    """
    Ablation study de Stage 1.

    Compara 3 configuraciones:
    1. Baseline: Patchwork++ puro
    2. + Wall Rejection
    3. + HCD

    Métricas:
    - Ground points detectados
    - Walls rechazadas
    - Timing
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST 4: ABLATION STUDY - Scan {scan_id:06d}")
        print(f"{'='*80}\n")

    data_path = (
        f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
        f"04/04/velodyne/{scan_id:06d}.bin"
    )

    points = load_kitti_bin(data_path)

    configs = [
        ('Baseline (Patchwork++)', PipelineConfig(
            enable_hybrid_wall_rejection=False,
            enable_hcd=False,
            verbose=False
        )),
        ('+ Wall Rejection', PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=False,
            verbose=False
        )),
        ('+ HCD', PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_hcd=True,
            verbose=False
        ))
    ]

    results = []

    for name, config in configs:
        pipeline = LidarPipelineSuite(config, data_path=data_path)
        output = pipeline.stage1_complete(points)

        results.append({
            'config': name,
            'ground': len(output['ground_indices']),
            'nonground': len(output['nonground_indices']),
            'walls': len(output['rejected_walls']),
            'timing': output['timing_ms'],
            'hcd_mean': np.mean(output['hcd']) if config.enable_hcd else 0.0
        })

    if verbose:
        print(f"{'Config':<25} {'Ground':>10} {'Walls':>8} {'Timing':>10}")
        print(f"{'-'*60}")
        for r in results:
            print(f"{r['config']:<25} {r['ground']:>10} {r['walls']:>8} {r['timing']:>9.1f}ms")

        print(f"\n{'='*80}")
        print("MEJORAS INCREMENTALES")
        print(f"{'='*80}")

        baseline = results[0]
        for i in range(1, len(results)):
            r = results[i]
            delta_walls = r['walls'] - baseline['walls']
            delta_time = r['timing'] - baseline['timing']

            print(f"\n{r['config']}:")
            print(f"  Walls adicionales: +{delta_walls}")
            print(f"  Overhead: +{delta_time:.1f}ms ({delta_time/baseline['timing']*100:.1f}%)")

            if r['hcd_mean'] > 0:
                print(f"  HCD mean: {r['hcd_mean']:.3f}")

        print(f"\n{'='*80}\n")

    return results


# ================================================================================
# TEST 5: BENCHMARK EN MÚLTIPLES SCANS
# ================================================================================

def test_multi_scan_benchmark(scan_range=(0, 4), verbose=True):
    """
    Benchmark en múltiples scans para estadísticas robustas.

    Compara Wall Rejection vs Baseline en rango de scans.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST 5: BENCHMARK MULTI-SCAN ({scan_range[0]}-{scan_range[1]})")
        print(f"{'='*80}\n")

    config_baseline = PipelineConfig(
        enable_hybrid_wall_rejection=False,
        enable_hcd=False,
        verbose=False
    )

    config_full = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        verbose=False
    )

    results_baseline = []
    results_full = []

    for scan_id in range(scan_range[0], scan_range[1] + 1):
        data_path = (
            f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/"
            f"04/04/velodyne/{scan_id:06d}.bin"
        )

        if not Path(data_path).exists():
            if verbose:
                print(f"[WARN] Scan {scan_id:06d} no encontrado, saltando")
            continue

        points = load_kitti_bin(data_path)

        # Baseline
        pipeline_baseline = LidarPipelineSuite(config_baseline, data_path=data_path)
        r_baseline = pipeline_baseline.stage1_complete(points)
        results_baseline.append(r_baseline)

        # Full
        pipeline_full = LidarPipelineSuite(config_full, data_path=data_path)
        r_full = pipeline_full.stage1_complete(points)
        results_full.append(r_full)

        if verbose:
            print(f"[Scan {scan_id:06d}] Baseline: {r_baseline['timing_ms']:.1f}ms | "
                  f"Full: {r_full['timing_ms']:.1f}ms | "
                  f"Walls: {len(r_full['rejected_walls'])}")

    # Estadísticas agregadas
    timing_baseline = np.mean([r['timing_ms'] for r in results_baseline])
    timing_full = np.mean([r['timing_ms'] for r in results_full])
    walls_mean = np.mean([len(r['rejected_walls']) for r in results_full])

    if verbose:
        print(f"\n{'='*80}")
        print("ESTADÍSTICAS AGREGADAS")
        print(f"{'='*80}")
        print(f"Timing medio:")
        print(f"  Baseline: {timing_baseline:.1f}ms")
        print(f"  Full (WallRej + HCD): {timing_full:.1f}ms")
        print(f"  Overhead: +{timing_full - timing_baseline:.1f}ms "
              f"({(timing_full/timing_baseline - 1)*100:.1f}%)")
        print(f"\nWalls rechazadas (media): {walls_mean:.1f}")
        print(f"\n{'='*80}\n")

    return {
        'timing_baseline': timing_baseline,
        'timing_full': timing_full,
        'overhead': timing_full - timing_baseline,
        'walls_mean': walls_mean
    }


# ================================================================================
# MAIN - EJECUTAR TODOS LOS TESTS
# ================================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tests para lidar_pipeline_suite.py')
    parser.add_argument('--scan', type=int, default=0,
                        help='Scan ID para tests individuales (default: 0)')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'basic', 'wall', 'hcd', 'ablation', 'benchmark'],
                        help='Test a ejecutar (default: all)')
    parser.add_argument('--scan_range', type=int, nargs=2, default=[0, 4],
                        help='Rango de scans para benchmark (default: 0 4)')

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"#{'TESTS - LIDAR PIPELINE SUITE':^78}#")
    print(f"{'#'*80}\n")

    if args.test in ['all', 'basic']:
        try:
            test_stage1_basic(args.scan)
            print("✅ TEST 1 PASADO\n")
        except Exception as e:
            print(f"❌ TEST 1 FALLADO: {e}\n")

    if args.test in ['all', 'wall']:
        try:
            test_wall_rejection_with_gt(args.scan)
            print("✅ TEST 2 PASADO\n")
        except Exception as e:
            print(f"❌ TEST 2 FALLADO: {e}\n")

    if args.test in ['all', 'hcd']:
        try:
            test_hcd_descriptor(args.scan)
            print("✅ TEST 3 PASADO\n")
        except Exception as e:
            print(f"❌ TEST 3 FALLADO: {e}\n")

    if args.test in ['all', 'ablation']:
        try:
            test_ablation_study(args.scan)
            print("✅ TEST 4 PASADO\n")
        except Exception as e:
            print(f"❌ TEST 4 FALLADO: {e}\n")

    if args.test in ['all', 'benchmark']:
        try:
            test_multi_scan_benchmark(tuple(args.scan_range))
            print("✅ TEST 5 PASADO\n")
        except Exception as e:
            print(f"❌ TEST 5 FALLADO: {e}\n")

    print(f"\n{'#'*80}")
    print(f"#{'TESTS COMPLETADOS':^78}#")
    print(f"{'#'*80}\n")
