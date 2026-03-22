#!/usr/bin/env python3
"""
Evaluación unificada de predicciones .label (SemanticKITTI format).
Compara cualquier método DL contra ground truth usando métricas binarias
(obstáculo vs ground), ignorando labels 0, 1, 52, 99.

Uso:
    python3 eval_predictions.py \
        --pred_dir /path/to/predictions \
        --gt_dir /path/to/labels \
        --stride 1 \
        --method "Cylinder3D"
"""
import argparse
import os
import numpy as np
from pathlib import Path

# === Labels SemanticKITTI ===
OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20,     # vehículos
    30, 31, 32,                       # personas
    50, 51,                           # estructuras
    70, 71,                           # vegetación
    80, 81,                           # troncos/postes
    252, 253, 254, 255, 256, 257, 258, 259  # moving
], dtype=np.uint32)

GROUND_LABELS = np.array([40, 44, 48, 49, 60, 72], dtype=np.uint32)

IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)

# Mapping de learning_map para predicciones DL (clase → label original)
# Cylinder3D predice clases 0-19, hay que mapearlas a labels originales
LEARNING_MAP_INV = {
    0: 0,       # unlabeled
    1: 10,      # car
    2: 11,      # bicycle
    3: 15,      # motorcycle
    4: 18,      # truck
    5: 20,      # other-vehicle
    6: 30,      # person
    7: 31,      # bicyclist
    8: 32,      # motorcyclist
    9: 40,      # road
    10: 44,     # parking
    11: 48,     # sidewalk
    12: 49,     # other-ground
    13: 50,     # building
    14: 51,     # fence
    15: 70,     # vegetation
    16: 71,     # trunk
    17: 72,     # terrain
    18: 80,     # pole
    19: 81,     # traffic-sign
}


def evaluate(pred_dir, gt_dir, stride=1, method="Unknown", raw_labels=False):
    """
    Evalúa predicciones vs GT en formato binario (obstáculo vs no obstáculo).

    Args:
        pred_dir: directorio con archivos .label de predicciones
        gt_dir: directorio con archivos .label de ground truth
        stride: evaluar cada N frames
        method: nombre del método para el informe
        raw_labels: si True, las predicciones ya usan labels originales (no clases 0-19)
    """
    pred_files = sorted(Path(pred_dir).glob("*.label"))
    gt_files = sorted(Path(gt_dir).glob("*.label"))

    if not pred_files:
        print(f"ERROR: No se encontraron archivos .label en {pred_dir}")
        return
    if not gt_files:
        print(f"ERROR: No se encontraron archivos .label en {gt_dir}")
        return

    # Filtrar por stride
    pred_files = pred_files[::stride]
    gt_files = gt_files[::stride]

    # Verificar que coinciden
    pred_ids = {f.stem for f in pred_files}
    gt_ids = {f.stem for f in gt_files}
    common = sorted(pred_ids & gt_ids)

    if len(common) == 0:
        print("ERROR: No hay frames comunes entre predicciones y GT")
        return

    print(f"{'='*80}")
    print(f"EVALUACIÓN: {method}")
    print(f"{'='*80}")
    print(f"  Predicciones: {pred_dir}")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Frames: {len(common)} (stride={stride})")
    print()

    tp_total, fp_total, fn_total = 0, 0, 0
    n_points_total = 0
    n_valid_total = 0

    # Métricas por clase (recall)
    class_names = {
        10: 'car', 11: 'bicycle', 13: 'bus', 15: 'motorcycle',
        16: 'on-rails', 18: 'truck', 20: 'other-veh',
        30: 'person', 31: 'bicyclist', 32: 'motorcyclist',
        50: 'building', 51: 'fence',
        70: 'vegetation', 71: 'trunk',
        80: 'pole', 81: 'sign',
        252: 'mov-car', 253: 'mov-bicyclist', 254: 'mov-person',
        255: 'mov-motorcyclist', 256: 'mov-on-rails', 257: 'mov-bus',
        258: 'mov-truck', 259: 'mov-other-veh'
    }
    class_tp = {l: 0 for l in OBSTACLE_LABELS}
    class_total = {l: 0 for l in OBSTACLE_LABELS}

    # Métricas por distancia
    dist_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80)]
    dist_tp = {r: 0 for r in dist_ranges}
    dist_fp = {r: 0 for r in dist_ranges}
    dist_fn = {r: 0 for r in dist_ranges}

    for i, scan_id in enumerate(common):
        pred_file = Path(pred_dir) / f"{scan_id}.label"
        gt_file = Path(gt_dir) / f"{scan_id}.label"

        # Cargar predicciones
        pred_raw = np.fromfile(str(pred_file), dtype=np.uint32)
        pred_labels = pred_raw & 0xFFFF

        # Si las predicciones son clases 0-19, mapear a labels originales
        if not raw_labels:
            pred_mapped = np.zeros_like(pred_labels)
            for cls, label in LEARNING_MAP_INV.items():
                pred_mapped[pred_labels == cls] = label
            pred_labels = pred_mapped

        # Cargar GT
        gt_raw = np.fromfile(str(gt_file), dtype=np.uint32)
        gt_labels = gt_raw & 0xFFFF

        # Cargar puntos para distancia
        velodyne_dir = Path(gt_dir).parent / "velodyne"
        bin_file = velodyne_dir / f"{scan_id}.bin"
        if bin_file.exists():
            points = np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4)
            dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        else:
            dist = None

        # Verificar longitudes
        n = min(len(pred_labels), len(gt_labels))
        pred_labels = pred_labels[:n]
        gt_labels = gt_labels[:n]

        # Máscaras
        valid_mask = ~np.isin(gt_labels, IGNORE_LABELS)
        gt_obs = np.isin(gt_labels, OBSTACLE_LABELS) & valid_mask
        pred_obs = np.isin(pred_labels, OBSTACLE_LABELS) & valid_mask

        # Métricas globales
        tp = int(np.sum(gt_obs & pred_obs))
        fp = int(np.sum(~gt_obs & pred_obs))
        fn = int(np.sum(gt_obs & ~pred_obs))

        tp_total += tp
        fp_total += fp
        fn_total += fn
        n_points_total += n
        n_valid_total += int(np.sum(valid_mask))

        # Métricas por clase
        for label in OBSTACLE_LABELS:
            mask_class = (gt_labels == label) & valid_mask
            n_class = int(np.sum(mask_class))
            if n_class > 0:
                class_total[label] += n_class
                class_tp[label] += int(np.sum(mask_class & pred_obs))

        # Métricas por distancia
        if dist is not None:
            dist = dist[:n]
            for (d_min, d_max) in dist_ranges:
                d_mask = (dist >= d_min) & (dist < d_max) & valid_mask
                gt_d = gt_obs & d_mask
                pred_d = pred_obs & d_mask
                dist_tp[(d_min, d_max)] += int(np.sum(gt_d & pred_d))
                dist_fp[(d_min, d_max)] += int(np.sum(~gt_d & pred_d & d_mask))
                dist_fn[(d_min, d_max)] += int(np.sum(gt_d & ~pred_d))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(common)} frames procesados...")

    # === RESULTADOS ===
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0

    print(f"\n{'='*80}")
    print(f"RESULTADOS — {method}")
    print(f"{'='*80}")
    print(f"  Puntos totales: {n_points_total:,}")
    print(f"  Puntos válidos: {n_valid_total:,} ({100*n_valid_total/n_points_total:.1f}%)")
    print(f"  TP: {tp_total:,}  FP: {fp_total:,}  FN: {fn_total:,}")
    print(f"\n  F1:        {100*f1:.2f}%")
    print(f"  IoU:       {100*iou:.2f}%")
    print(f"  Precision: {100*precision:.2f}%")
    print(f"  Recall:    {100*recall:.2f}%")

    # Por clase
    print(f"\n{'='*80}")
    print(f"RECALL POR CLASE")
    print(f"{'='*80}")
    print(f"  {'Clase':<16} {'N puntos':>10}   {'Recall':>8}")
    print(f"  {'-'*40}")
    for label in sorted(class_total.keys(), key=lambda l: class_total[l], reverse=True):
        n = class_total[label]
        if n > 0:
            r = class_tp[label] / n
            name = class_names.get(label, f"label-{label}")
            print(f"  {name:<16} {n:>10,}   {100*r:>7.2f}%")

    # Por distancia
    if dist is not None:
        print(f"\n{'='*80}")
        print(f"F1 POR DISTANCIA")
        print(f"{'='*80}")
        print(f"  {'Rango':<10} {'F1':>8}   {'P':>8}   {'R':>8}")
        print(f"  {'-'*40}")
        for (d_min, d_max) in dist_ranges:
            tp_d = dist_tp[(d_min, d_max)]
            fp_d = dist_fp[(d_min, d_max)]
            fn_d = dist_fn[(d_min, d_max)]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0
            print(f"  {d_min}-{d_max}m     {100*f1_d:>7.2f}%  {100*p_d:>7.2f}%  {100*r_d:>7.2f}%")

    return {
        'method': method,
        'f1': f1, 'iou': iou, 'precision': precision, 'recall': recall,
        'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar predicciones .label en formato SemanticKITTI')
    parser.add_argument('--pred_dir', required=True, help='Directorio con predicciones .label')
    parser.add_argument('--gt_dir', required=True, help='Directorio con GT .label')
    parser.add_argument('--stride', type=int, default=1, help='Evaluar cada N frames')
    parser.add_argument('--method', type=str, default='Unknown', help='Nombre del método')
    parser.add_argument('--raw_labels', action='store_true',
                        help='Las predicciones usan labels originales (no clases 0-19)')
    args = parser.parse_args()

    evaluate(args.pred_dir, args.gt_dir, args.stride, args.method, args.raw_labels)
