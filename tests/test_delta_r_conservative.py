#!/usr/bin/env python3
"""
Test comparativo: Delta-r original vs Delta-r conservador vs Stage 1 solo.

Compara tres configuraciones:
1. Stage 1 solo (PW++ + WR) — baseline
2. Stage 1 + delta-r original — modo actual
3. Stage 1 + delta-r conservador — solo rescate en bins fiables

Secuencias: KITTI 00 y 04 (datos locales).
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file

# ========================================
# ETIQUETAS SemanticKITTI (corregidas)
# ========================================

OBSTACLE_LABELS = np.array([
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    50, 51,
    70, 71,
    80, 81,
    252, 253, 254, 255, 256, 257, 258, 259
], dtype=np.uint32)

IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)


def load_scan(scan_id, seq):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    gt_mask = np.isin(semantic_labels, OBSTACLE_LABELS)
    valid_mask = ~np.isin(semantic_labels, IGNORE_LABELS)
    return points, gt_mask, valid_mask


def compute_metrics(gt_mask, pred_mask, valid_mask):
    gt_v = gt_mask & valid_mask
    pred_v = pred_mask & valid_mask
    tp = int(np.sum(gt_v & pred_v))
    fp = int(np.sum((~gt_v) & pred_v))
    fn = int(np.sum(gt_v & (~pred_v)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn}


def test_sequence(seq, stride=10, max_frames=50):
    info = get_sequence_info(seq)
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    # Listar scans disponibles
    velodyne_dir = info['data_dir'] / 'velodyne'
    scan_files = sorted(velodyne_dir.glob('*.bin'))
    scan_ids = [int(f.stem) for f in scan_files]
    scan_ids = scan_ids[::stride][:max_frames]

    print(f"\nSecuencia {seq}: {len(scan_ids)} frames (stride={stride})")
    print("=" * 100)

    # Configuraciones a comparar
    configs = {
        'Stage 1 solo (baseline)': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            verbose=False
        ),
        'Stage 1 + delta-r original': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            delta_r_conservative=False,
            verbose=False
        ),
        'Stage 1 + delta-r conservador': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            delta_r_conservative=True,
            delta_r_min_nz=0.95,
            verbose=False
        ),
        'Stage 1 + delta-r cons. sin voids': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            delta_r_conservative=True,
            delta_r_min_nz=0.95,
            delta_r_rescue_voids=False,
            verbose=False
        ),
    }

    results = {}

    for config_name, config in configs.items():
        pipeline = LidarPipelineSuite(config)
        total_tp, total_fp, total_fn = 0, 0, 0
        total_time = 0.0
        n_rescued_total = 0

        for scan_id in scan_ids:
            pts, gt_mask, valid_mask = load_scan(scan_id, seq)

            t0 = time.time()

            if config_name == 'Stage 1 solo (baseline)':
                # Solo Stage 1
                s1 = pipeline.stage1_complete(pts)
                pred_mask = np.zeros(len(pts), dtype=bool)
                pred_mask[s1['nonground_indices']] = True
            else:
                # Stage 1 + Stage 2 (delta-r)
                result = pipeline.stage2_complete(pts)
                pred_mask = result['obs_mask']

            elapsed = time.time() - t0
            total_time += elapsed

            m = compute_metrics(gt_mask, pred_mask, valid_mask)
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']

        # Métricas globales
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        avg_ms = (total_time / len(scan_ids)) * 1000

        results[config_name] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'avg_ms': avg_ms
        }

    # Tabla de resultados
    print(f"\n{'Config':<35} {'F1':>8} {'IoU':>8} {'P':>8} {'R':>8} {'FP':>10} {'FN':>10} {'ms/frame':>10}")
    print("-" * 107)

    baseline_f1 = results['Stage 1 solo (baseline)']['f1']

    for name, m in results.items():
        delta = m['f1'] - baseline_f1
        delta_str = f"({delta:+.2%})" if name != 'Stage 1 solo (baseline)' else ""
        print(f"{name:<35} {m['f1']:>7.2%} {m['iou']:>7.2%} {m['precision']:>7.2%} {m['recall']:>7.2%} "
              f"{m['fp']:>10} {m['fn']:>10} {m['avg_ms']:>9.1f} {delta_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Delta-r conservador vs original')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--max_frames', type=int, default=50)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    args = parser.parse_args()

    print("=" * 100)
    print("TEST: Delta-r original vs Delta-r conservador vs Stage 1 solo")
    print("=" * 100)

    all_results = {}

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', stride=args.stride, max_frames=args.max_frames)

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', stride=args.stride, max_frames=args.max_frames)

    # Resumen global
    if len(all_results) > 1 and all(v is not None for v in all_results.values()):
        print(f"\n{'='*100}")
        print("RESUMEN GLOBAL")
        print(f"{'='*100}")

        config_names = list(all_results['00'].keys())
        print(f"\n{'Config':<35} {'Seq 00 F1':>10} {'Seq 04 F1':>10} {'Media F1':>10}")
        print("-" * 75)
        for name in config_names:
            f1_00 = all_results['00'][name]['f1']
            f1_04 = all_results['04'][name]['f1']
            f1_avg = (f1_00 + f1_04) / 2
            print(f"{name:<35} {f1_00:>9.2%} {f1_04:>9.2%} {f1_avg:>9.2%}")


if __name__ == '__main__':
    main()
