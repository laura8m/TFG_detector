#!/usr/bin/env python3
"""
Evaluador de range_projection.py con métricas de detección.

Modifica temporalmente range_projection.py para agregar logging de métricas,
ejecuta el pipeline, y analiza resultados.

Uso:
    python3 evaluate_range_projection.py --scan_start 0 --scan_end 0
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse

# Agregar paths
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)


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


def extract_detections_from_range_projection(scan_id):
    """
    Ejecuta range_projection.py internamente y extrae máscaras de detección.

    NOTA: Esta función requiere modificar range_projection.py para exportar
    la máscara de detección final. Por ahora, retorna None.
    """
    # TODO: Implementar extracción de resultados
    # Opciones:
    # 1. Modificar range_projection.py para guardar máscara en archivo .npy
    # 2. Usar ROS 2 bags para capturar mensajes publicados
    # 3. Ejecutar como módulo Python y acceder a atributos del nodo

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--scan_end', type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUACIÓN DE RANGE_PROJECTION.PY")
    print("=" * 80)
    print(f"Scan range: {args.scan_start} - {args.scan_end}")
    print()

    print("[INFO] Estrategia de evaluación:")
    print("  1. Leer código de range_projection.py para entender algoritmo")
    print("  2. Comparar con lidar_pipeline_suite.py")
    print("  3. Analizar diferencias clave")
    print()

    # Cargar ground truth para análisis
    for scan_id in range(args.scan_start, args.scan_end + 1):
        print(f"\n{'='*80}")
        print(f"Scan {scan_id}")
        print(f"{'='*80}")

        points, semantic_labels = load_kitti_scan(scan_id)
        gt_masks = get_ground_truth_masks(semantic_labels)

        print(f"✓ Datos cargados: {len(points)} puntos")
        print(f"  GT obstacles: {gt_masks['obstacle'].sum()} ({100*gt_masks['obstacle'].sum()/len(points):.1f}%)")
        print(f"  GT ground: {gt_masks['ground'].sum()} ({100*gt_masks['ground'].sum()/len(points):.1f}%)")
        print()

        # Intentar extraer detecciones de range_projection
        pred_mask = extract_detections_from_range_projection(scan_id)

        if pred_mask is not None:
            metrics = compute_detection_metrics(gt_masks['obstacle'], pred_mask)
            print(f"✓ Métricas de detección:")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
            print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
        else:
            print("[WARN] No se pudieron extraer detecciones de range_projection.py")
            print("[INFO] Para obtener métricas, necesitas:")
            print("  - Modificar range_projection.py para guardar belief_prob como .npy")
            print("  - O usar ROS 2 bags para capturar /bayes_cloud")
            print("  - O ejecutar como módulo Python (requiere configuración ROS 2)")

    print()
    print("=" * 80)
    print("ANÁLISIS CONCEPTUAL")
    print("=" * 80)
    print()
    print("Según el análisis del código de range_projection.py (líneas 998-1218):")
    print()
    print("1. PROBABILIDAD BINARIA (línea 998):")
    print("   raw_probability = (range_image < threshold_obs).astype(np.float32)")
    print("   → Usa threshold simple: delta_r < -0.3 → P=1.0 (obstacle)")
    print()
    print("2. BAYES FILTER (líneas 1185-1218):")
    print("   - Ecuación 9 de Dewan (igual que lidar_pipeline_suite.py)")
    print("   - l_t = log(P/(1-P)) + l_{t-1} - l_0")
    print("   - Clamp: [-2.5, 2.5] (más restrictivo que nuestro [-10, 10])")
    print()
    print("3. PROYECCIÓN A RANGE IMAGE:")
    print("   - Línea 590: self.range_image[u_sorted, v_sorted] = delta_sorted")
    print("   - Estrategia: CLOSEST WINS (último write gana = más cercano)")
    print("   - Esto es IDÉNTICO a nuestra implementación original")
    print()
    print("PREDICCIÓN:")
    print("  Si range_projection.py usa CLOSEST WINS + PROBABILIDAD BINARIA,")
    print("  debería tener el MISMO problema de recall 43% que lidar_pipeline_suite.py")
    print()
    print("VERIFICACIÓN RECOMENDADA:")
    print("  1. Ejecutar range_projection.py con RViz:")
    print("     ./launch_range_projection.sh 0 0")
    print("  2. Observar /bayes_cloud en RViz")
    print("  3. Comparar visualmente con /cluster_points de lidar_pipeline_suite.py")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
