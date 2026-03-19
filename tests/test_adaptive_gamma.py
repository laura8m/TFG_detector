#!/usr/bin/env python3
"""
Test: Gamma Adaptativo + Warping Quality Check en Stage 3

Compara pipeline completo (Stages 2→3→4→5) con:
- Config A: gamma fijo 0.7 (actual)
- Config B: gamma adaptativo + warping quality check (nuevo)

Objetivo: Verificar que en seq 04 (highway, alta velocidad) el gamma adaptativo
reduce FP sin perjudicar seq 00 (urbano, baja velocidad).
"""

import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages')

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# ========================================
# UTILIDADES (copiadas de test_stage4)
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
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def get_gt_obstacle_mask(semantic_labels):
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20,
        30, 31, 32,
        50, 51, 52,
        70, 71,
        80, 81,
        99,
        252, 253, 254, 255, 256, 257, 258, 259
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


# ========================================
# TEST POR SECUENCIA
# ========================================

def run_pipeline(seq, scan_start, n_frames, config, pipeline_name, verbose_frames=False):
    """Ejecutar pipeline completo y devolver métricas del último frame"""
    info = SEQUENCES[seq]
    poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])

    pipeline = LidarPipelineSuite(config)

    for i in range(n_frames):
        scan_id = scan_start + i
        pts, _ = load_kitti_scan(scan_id, seq)
        delta_pose = None if i == 0 else LidarPipelineSuite.compute_delta_pose(
            poses[scan_start + i - 1], poses[scan_start + i]
        )
        result = pipeline.stage5_per_point(pts, delta_pose=delta_pose)

        if verbose_frames and (i + 1) % 5 == 0:
            ego_speed = result.get('ego_speed', 0)
            gamma_eff = result.get('gamma_effective', config.gamma)
            assoc = result.get('warp_association_ratio', 0)
            print(f"    Frame {scan_id}: {result['obs_mask'].sum()} obs | speed={ego_speed:.3f} m/f | gamma={gamma_eff:.3f} | assoc={100*assoc:.0f}%")

    # Métricas del último frame
    scan_ref = scan_start + n_frames - 1
    _, labels_ref = load_kitti_scan(scan_ref, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)
    metrics = compute_detection_metrics(gt_mask, result['obs_mask'])

    print(f"  {pipeline_name}: P={100*metrics['precision']:.1f}% R={100*metrics['recall']:.1f}% F1={100*metrics['f1']:.1f}% | FP={metrics['fp']} FN={metrics['fn']}")

    return metrics, result


def test_sequence(seq, scan_start, n_frames):
    info = SEQUENCES[seq]
    if not info['data_dir'].exists():
        print(f"  [SKIP] Datos no encontrados: {info['data_dir']}")
        return None

    print("=" * 80)
    print(f"SECUENCIA {seq} | Frames {scan_start}-{scan_start + n_frames - 1}")
    print("=" * 80)

    results = {}

    # --- Stage 2 baseline ---
    print("\n--- Stage 2 solo (baseline) ---")
    pipeline_s2 = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True, enable_hcd=True, verbose=False
    ))
    scan_ref = scan_start + n_frames - 1
    pts_ref, labels_ref = load_kitti_scan(scan_ref, seq)
    gt_mask = get_gt_obstacle_mask(labels_ref)
    r_s2 = pipeline_s2.stage2_complete(pts_ref)
    m_s2 = compute_detection_metrics(gt_mask, r_s2['obs_mask'])
    print(f"  Stage 2:       P={100*m_s2['precision']:.1f}% R={100*m_s2['recall']:.1f}% F1={100*m_s2['f1']:.1f}% | FP={m_s2['fp']} FN={m_s2['fn']}")
    results['stage2'] = m_s2

    # --- Config A: gamma fijo (pipeline actual) ---
    print(f"\n--- Pipeline completo: gamma FIJO 0.7 ---")
    config_fixed = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        enable_cluster_filtering=True,
        gamma=0.7,
        gamma_min=0.7,  # Desactivar adaptativo: gamma_min = gamma (nunca baja)
        warp_min_association=0.0,  # Desactivar quality check
        cluster_min_pts=15,
        verbose=False
    )
    m_fixed, r_fixed = run_pipeline(seq, scan_start, n_frames, config_fixed, "Gamma fijo 0.7", verbose_frames=True)
    results['gamma_fijo'] = m_fixed

    # --- Config B: gamma adaptativo (nuevo) ---
    print(f"\n--- Pipeline completo: gamma ADAPTATIVO (thr=0.3, scale=1.5) ---")
    config_adaptive = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        enable_cluster_filtering=True,
        gamma=0.7,
        gamma_min=0.0,
        gamma_speed_threshold=0.3,
        gamma_speed_scale=1.5,
        warp_min_association=0.3,
        cluster_min_pts=15,
        verbose=False
    )
    m_adapt, r_adapt = run_pipeline(seq, scan_start, n_frames, config_adaptive, "Adapt(0.3,1.5)", verbose_frames=True)
    results['adapt_0.3_1.5'] = m_adapt

    # --- Config C: gamma adaptativo con threshold más alto ---
    print(f"\n--- Pipeline completo: gamma ADAPTATIVO (thr=0.8, scale=2.0) ---")
    config_adaptive2 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        enable_cluster_filtering=True,
        gamma=0.7,
        gamma_min=0.0,
        gamma_speed_threshold=0.8,
        gamma_speed_scale=2.0,
        warp_min_association=0.3,
        cluster_min_pts=15,
        verbose=False
    )
    m_adapt2, r_adapt2 = run_pipeline(seq, scan_start, n_frames, config_adaptive2, "Adapt(0.8,2.0)", verbose_frames=True)
    results['adapt_0.8_2.0'] = m_adapt2

    # --- Config D: gamma adaptativo conservador ---
    print(f"\n--- Pipeline completo: gamma ADAPTATIVO (thr=1.0, scale=2.0, min=0.2) ---")
    config_adaptive3 = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=True,
        enable_shadow_validation=True,
        enable_cluster_filtering=True,
        gamma=0.7,
        gamma_min=0.2,
        gamma_speed_threshold=1.0,
        gamma_speed_scale=2.0,
        warp_min_association=0.3,
        cluster_min_pts=15,
        verbose=False
    )
    m_adapt3, r_adapt3 = run_pipeline(seq, scan_start, n_frames, config_adaptive3, "Adapt(1.0,2.0,0.2)", verbose_frames=True)
    results['adapt_1.0_2.0'] = m_adapt3

    # --- Comparativa ---
    print(f"\n{'='*60}")
    print(f"COMPARATIVA SECUENCIA {seq}")
    print(f"{'='*60}")
    print(f"\n{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8}")
    print("-" * 71)
    for name, m in results.items():
        print(f"{name:<25} {100*m['precision']:>9.1f}% {100*m['recall']:>9.1f}% {100*m['f1']:>9.1f}% {m['fp']:>8} {m['fn']:>8}")

    # Impacto
    if 'gamma_fijo' in results and 'gamma_adapt' in results:
        fp_change = results['gamma_fijo']['fp'] - results['gamma_adapt']['fp']
        recall_change = 100 * (results['gamma_adapt']['recall'] - results['gamma_fijo']['recall'])
        f1_change = 100 * (results['gamma_adapt']['f1'] - results['gamma_fijo']['f1'])
        print(f"\n  IMPACTO gamma adaptativo vs fijo:")
        print(f"    FP: {fp_change:+d} ({'menos' if fp_change > 0 else 'más'})")
        print(f"    Recall: {recall_change:+.2f}%")
        print(f"    F1: {f1_change:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Adaptive Gamma')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    args = parser.parse_args()

    all_results = {}

    if args.seq in ('04', 'both'):
        all_results['04'] = test_sequence('04', args.scan_start, args.n_frames)

    if args.seq in ('00', 'both'):
        all_results['00'] = test_sequence('00', args.scan_start, args.n_frames)

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("RESUMEN GLOBAL")
        print("=" * 80)
        for seq, res in all_results.items():
            if res is None:
                continue
            print(f"\n  Secuencia {seq}:")
            for name, m in res.items():
                print(f"    {name:<25} F1={100*m['f1']:.1f}%  P={100*m['precision']:.1f}%  R={100*m['recall']:.1f}%")


if __name__ == '__main__':
    main()
