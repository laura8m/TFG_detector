#!/usr/bin/env python3
"""
Test completo de Stages 1, 2 y 3 en KITTI Sequences 00 y 04.

Ejecuta:
- Stage 1: Patchwork++ ground segmentation (analisis de wall misclassification)
- Stage 2: Delta-r anomaly detection (single frame)
- Stage 3: DBSCAN cluster filtering (pipeline completo Stage 1+2+3)

Genera metricas contra ground truth de SemanticKITTI.
"""

import sys
import numpy as np
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
import pypatchworkpp
from scipy.spatial import cKDTree

# ========================================
# CONSTANTES
# ========================================

LABEL_NAMES = {
    0:'unlabeled', 1:'outlier', 10:'car', 11:'bicycle', 13:'bus',
    15:'motorcycle', 16:'on-rails', 18:'truck', 20:'other-vehicle',
    30:'person', 31:'bicyclist', 32:'motorcyclist',
    40:'road', 44:'parking', 48:'sidewalk', 49:'other-ground',
    50:'building', 51:'fence', 52:'other-structure', 60:'lane-marking',
    70:'vegetation', 71:'trunk', 72:'terrain',
    80:'pole', 81:'traffic-sign', 99:'other-object',
    252:'moving-car', 253:'moving-bicyclist', 254:'moving-person',
    255:'moving-motorcyclist', 256:'moving-on-rails', 257:'moving-bus',
    258:'moving-truck', 259:'moving-other-vehicle',
}

GROUND_LABELS = {40, 44, 48, 49, 60, 72}
OBSTACLE_LABELS = {10,11,13,15,16,18,20,30,31,32,50,51,52,70,71,80,81,99,
                   252,253,254,255,256,257,258,259}
CRITICAL_LABELS = {10, 13, 15, 18, 20, 30, 31, 32, 252, 253, 254, 255, 257, 258, 259}

from data_paths import get_sequence_info, get_scan_file, get_label_file, get_velodyne_dir, get_labels_dir, get_poses_file

SEQ_DESCRIPTIONS = {
    '00': 'Urbana (edificios, coches, vegetacion, aceras)',
    '04': 'Autopista (vehiculos rapidos, vallas, vegetacion distante)',
}

def _build_sequences():
    seqs = {}
    for seq_id, desc in SEQ_DESCRIPTIONS.items():
        info = get_sequence_info(seq_id)
        seqs[seq_id] = {
            'velodyne_dir': str(get_velodyne_dir(seq_id)),
            'label_dir': str(info['label_dir']),
            'poses_file': info['poses_file'],
            'description': desc,
        }
    return seqs

SEQUENCES = _build_sequences()


# ========================================
# UTILIDADES
# ========================================

def load_scan(seq_id, scan_id):
    """Cargar scan y labels de KITTI"""
    seq = SEQUENCES[seq_id]
    scan_file = Path(seq['velodyne_dir']) / f"{scan_id:06d}.bin"
    label_file = Path(seq['label_dir']) / f"{scan_id:06d}.label"

    points = np.fromfile(str(scan_file), dtype=np.float32).reshape(-1, 4)[:, :3]

    if label_file.exists():
        labels = np.fromfile(str(label_file), dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)

    return points, semantic_labels


def get_gt_masks(semantic_labels):
    """Obtener mascaras de ground truth"""
    gt_ground = np.zeros(len(semantic_labels), dtype=bool)
    for l in GROUND_LABELS:
        gt_ground |= (semantic_labels == l)

    gt_obstacle = np.zeros(len(semantic_labels), dtype=bool)
    for l in OBSTACLE_LABELS:
        gt_obstacle |= (semantic_labels == l)

    gt_critical = np.zeros(len(semantic_labels), dtype=bool)
    for l in CRITICAL_LABELS:
        gt_critical |= (semantic_labels == l)

    return gt_ground, gt_obstacle, gt_critical


def compute_metrics(gt_mask, pred_mask):
    """Precision, Recall, F1"""
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# ========================================
# TEST STAGE 1: PATCHWORK++ GROUND SEGMENTATION
# ========================================

def test_stage1_patchwork(seq_id, scan_id=0):
    """Test Patchwork++ vanilla: que obstaculos clasifica como ground"""
    print(f"\n{'='*80}")
    print(f"STAGE 1: PATCHWORK++ - Sequence {seq_id}, Frame {scan_id}")
    print(f"{'='*80}")

    points, semantic_labels = load_scan(seq_id, scan_id)
    gt_ground, gt_obstacle, gt_critical = get_gt_masks(semantic_labels)

    print(f"Total puntos: {len(points)}")
    print(f"GT ground: {np.sum(gt_ground)} ({100*np.mean(gt_ground):.1f}%)")
    print(f"GT obstacle: {np.sum(gt_obstacle)} ({100*np.mean(gt_obstacle):.1f}%)")

    # Ejecutar Patchwork++
    params = pypatchworkpp.Parameters()
    params.verbose = False
    pwpp = pypatchworkpp.patchworkpp(params)
    pwpp.estimateGround(points)

    ground_pts = pwpp.getGround()
    nonground_pts = pwpp.getNonground()

    # Asociar con indices originales
    tree = cKDTree(points)
    _, pw_ground_idx = tree.query(ground_pts, k=1) if len(ground_pts) > 0 else (None, np.array([], dtype=int))
    pw_ground_mask = np.zeros(len(points), dtype=bool)
    pw_ground_mask[pw_ground_idx] = True

    print(f"\nPatchwork++ ground: {len(ground_pts)}, non-ground: {len(nonground_pts)}")

    # Obstaculos como ground
    misclass_mask = pw_ground_mask & gt_obstacle
    n_misclass = int(np.sum(misclass_mask))

    print(f"\nOBSTACULOS MAL CLASIFICADOS COMO GROUND: {n_misclass} / {np.sum(gt_obstacle)} ({100*n_misclass/max(1,np.sum(gt_obstacle)):.2f}%)")

    # Desglose por tipo
    misclass_by_type = {}
    misclass_labels = semantic_labels[np.where(misclass_mask)[0]]
    unique_labels, counts = np.unique(misclass_labels, return_counts=True)
    for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        name = LABEL_NAMES.get(lbl, f'unknown-{lbl}')
        total = int(np.sum(semantic_labels == lbl))
        misclass_by_type[name] = {'count': int(cnt), 'total': total, 'pct': 100*cnt/max(1,total)}
        print(f"  {name:<25s} {cnt:>6d} / {total:>6d} ({100*cnt/max(1,total):>5.1f}%)")

    # Normales por zona CZM
    normals = np.array(pwpp.getNormals())
    centers = np.array(pwpp.getCenters())
    czm_analysis = {}
    if len(centers) > 0:
        center_r = np.sqrt(centers[:,0]**2 + centers[:,1]**2)
        nz = np.abs(normals[:,2])
        print(f"\nPlanos CZM por zona:")
        for rmin, rmax, zname in [(0,9.64,'Z0'),(9.64,22.28,'Z1'),(22.28,48.56,'Z2'),(48.56,80,'Z3')]:
            in_zone = (center_r >= rmin) & (center_r < rmax)
            n_total = int(np.sum(in_zone))
            n_vert = int(np.sum(in_zone & (nz < 0.7)))
            pct = 100*n_vert/max(1,n_total)
            czm_analysis[zname] = {'total': n_total, 'vertical': n_vert, 'pct': pct}
            rvpf = " (RVPF activo)" if rmin == 0 else ""
            print(f"  {zname} [{rmin:.1f}-{rmax:.1f}m]{rvpf}: {n_total} planos, {n_vert} verticales ({pct:.1f}%)")

    # Analisis base vs entero
    base_analysis = {}
    print(f"\nAnalisis base vs objeto completo:")
    for label_id in [50, 51, 70, 10, 80, 71, 52, 99, 18, 13, 30]:
        name = LABEL_NAMES.get(label_id, str(label_id))
        obj_mask = semantic_labels == label_id
        n_total = int(np.sum(obj_mask))
        if n_total == 0: continue
        n_as_ground = int(np.sum(obj_mask & pw_ground_mask))
        if n_as_ground == 0: continue

        obj_z = points[obj_mask, 2]
        misclass_z = points[obj_mask & pw_ground_mask, 2]
        obj_z_range = obj_z.max() - obj_z.min()
        misclass_relative = float((misclass_z.mean() - obj_z.min()) / max(0.01, obj_z_range))
        pos = 'BASE' if misclass_relative < 0.3 else ('MEDIO' if misclass_relative < 0.7 else 'ALTO')

        misclass_r = np.sqrt(points[obj_mask & pw_ground_mask, 0]**2 + points[obj_mask & pw_ground_mask, 1]**2)
        n_z0 = int(np.sum(misclass_r <= 9.64))
        n_z123 = int(np.sum(misclass_r > 9.64))

        base_analysis[name] = {
            'n_as_ground': n_as_ground, 'n_total': n_total,
            'pct': 100*n_as_ground/n_total, 'position': pos,
            'relative_pct': 100*misclass_relative,
            'z_range': [float(misclass_z.min()), float(misclass_z.max())],
            'r_mean': float(misclass_r.mean()),
            'in_z0': n_z0, 'in_z123': n_z123
        }
        print(f"  {name:<20s} {n_as_ground:>5d}/{n_total:>5d} ({100*n_as_ground/n_total:>5.1f}%) pos={pos}({100*misclass_relative:.0f}%) r={misclass_r.mean():.1f}m z0={n_z0} z1-3={n_z123}")

    # Impacto navegacion
    n_critical_miss = int(np.sum(pw_ground_mask & gt_critical))
    print(f"\nObstaculos CRITICOS (vehiculos+personas) como ground: {n_critical_miss} / {np.sum(gt_critical)}")
    critical_detail = {}
    if n_critical_miss > 0:
        crit_labels = semantic_labels[np.where(pw_ground_mask & gt_critical)[0]]
        u, c = np.unique(crit_labels, return_counts=True)
        for i in np.argsort(-c):
            name = LABEL_NAMES.get(u[i], str(u[i]))
            critical_detail[name] = int(c[i])
            print(f"    {name}: {c[i]}")

    # Ground precision
    pw_correct = int(np.sum(pw_ground_mask & gt_ground))
    pw_total = int(np.sum(pw_ground_mask))

    return {
        'n_points': len(points),
        'gt_ground': int(np.sum(gt_ground)),
        'gt_obstacle': int(np.sum(gt_obstacle)),
        'pw_ground': len(ground_pts),
        'pw_nonground': len(nonground_pts),
        'n_misclassified': n_misclass,
        'misclass_pct': 100*n_misclass/max(1,int(np.sum(gt_obstacle))),
        'ground_precision': 100*pw_correct/max(1,pw_total),
        'misclass_by_type': misclass_by_type,
        'czm_analysis': czm_analysis,
        'base_analysis': base_analysis,
        'critical_miss': n_critical_miss,
        'critical_detail': critical_detail,
    }


# ========================================
# TEST STAGE 2: DELTA-R (SINGLE FRAME)
# ========================================

def test_stage2(seq_id, scan_id=0):
    """Test Stage 2 en un frame"""
    print(f"\n{'='*80}")
    print(f"STAGE 2: DELTA-R ANOMALY - Sequence {seq_id}, Frame {scan_id}")
    print(f"{'='*80}")

    points, semantic_labels = load_scan(seq_id, scan_id)
    gt_ground, gt_obstacle, gt_critical = get_gt_masks(semantic_labels)

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        verbose=False
    )
    pipeline = LidarPipelineSuite(config)
    result = pipeline.stage2_complete(points)

    metrics = compute_metrics(gt_obstacle, result['obs_mask'])

    print(f"Obstacles detectados: {np.sum(result['obs_mask'])}")
    print(f"GT obstacles: {np.sum(gt_obstacle)}")
    print(f"Precision: {100*metrics['precision']:.2f}%")
    print(f"Recall:    {100*metrics['recall']:.2f}%")
    print(f"F1:        {100*metrics['f1']:.2f}%")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Timing:    {result['timing_total_ms']:.0f} ms")

    # Analisis Stage 1 dentro del pipeline (con wall rejection)
    stage1_result = pipeline.stage1_complete(points)
    ground_mask_s1 = np.zeros(len(points), dtype=bool)
    ground_mask_s1[stage1_result['ground_indices']] = True
    n_ground_s1 = int(np.sum(ground_mask_s1))
    n_nonground_s1 = len(points) - n_ground_s1
    misclass_s1 = int(np.sum(ground_mask_s1 & gt_obstacle))

    print(f"\n[Stage 1 dentro del pipeline (con wall rejection)]:")
    print(f"  Ground: {n_ground_s1}, Non-ground: {n_nonground_s1}")
    print(f"  Obstaculos como ground: {misclass_s1} ({100*misclass_s1/max(1,int(np.sum(gt_obstacle))):.2f}%)")

    return {
        'n_obstacles_detected': int(np.sum(result['obs_mask'])),
        'gt_obstacles': int(np.sum(gt_obstacle)),
        'metrics': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        'timing_ms': float(result['timing_total_ms']),
        'stage1_ground': n_ground_s1,
        'stage1_misclass': misclass_s1,
        'stage1_misclass_pct': 100*misclass_s1/max(1,int(np.sum(gt_obstacle))),
    }


# ========================================
# TEST STAGE 3: DBSCAN CLUSTER FILTERING (PIPELINE COMPLETO)
# ========================================

def test_stage3(seq_id, scan_id=0):
    """Test Stage 3: pipeline completo Stage 1+2+3 (DBSCAN cluster filtering)"""
    print(f"\n{'='*80}")
    print(f"STAGE 3: DBSCAN CLUSTER FILTERING - Sequence {seq_id}, Frame {scan_id}")
    print(f"{'='*80}")

    points, semantic_labels = load_scan(seq_id, scan_id)
    gt_ground, gt_obstacle, gt_critical = get_gt_masks(semantic_labels)

    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        cluster_eps=0.5,
        cluster_min_samples=5,
        cluster_min_pts=15,
        verbose=False
    )
    pipeline = LidarPipelineSuite(config)
    result = pipeline.stage3_complete(points)

    metrics = compute_metrics(gt_obstacle, result['obs_mask'])

    print(f"Obstacles detectados: {np.sum(result['obs_mask'])}")
    print(f"GT obstacles: {np.sum(gt_obstacle)}")
    print(f"Precision: {100*metrics['precision']:.2f}%")
    print(f"Recall:    {100*metrics['recall']:.2f}%")
    print(f"F1:        {100*metrics['f1']:.2f}%")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Timing:    {result['timing_total_ms']:.0f} ms")

    n_clusters = result.get('n_clusters', 0)
    n_removed = result.get('n_cluster_total_removed', 0)
    print(f"Clusters: {n_clusters} valid | Removed: {n_removed} pts")

    return {
        'n_obstacles_detected': int(np.sum(result['obs_mask'])),
        'gt_obstacles': int(np.sum(gt_obstacle)),
        'metrics': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        'timing_ms': float(result['timing_total_ms']),
        'n_clusters': n_clusters,
        'n_removed': n_removed,
    }


# ========================================
# MAIN
# ========================================

def main():
    all_results = {}

    for seq_id in ['00', '04']:
        print(f"\n{'#'*80}")
        print(f"# SEQUENCE {seq_id} - {SEQUENCES[seq_id]['description']}")
        print(f"{'#'*80}")

        seq_results = {}

        # Stage 1: Patchwork++ vanilla
        seq_results['stage1'] = test_stage1_patchwork(seq_id, scan_id=0)

        # Stage 2: Delta-r (single frame)
        seq_results['stage2'] = test_stage2(seq_id, scan_id=0)

        # Stage 3: DBSCAN cluster filtering (pipeline completo)
        seq_results['stage3'] = test_stage3(seq_id, scan_id=0)

        all_results[seq_id] = seq_results

    # ========================================
    # TABLA COMPARATIVA FINAL
    # ========================================
    print(f"\n{'='*80}")
    print("TABLA COMPARATIVA FINAL - AMBAS SECUENCIAS")
    print(f"{'='*80}")

    for seq_id in ['00', '04']:
        r = all_results[seq_id]
        s2 = r['stage2']['metrics']
        s3 = r['stage3']['metrics']

        print(f"\n--- Sequence {seq_id} ({SEQUENCES[seq_id]['description']}) ---")
        print(f"{'Metrica':<20s} {'Stage 2':>12s} {'Stage 3':>12s}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")
        print(f"{'Precision':<20s} {100*s2['precision']:>11.2f}% {100*s3['precision']:>11.2f}%")
        print(f"{'Recall':<20s} {100*s2['recall']:>11.2f}% {100*s3['recall']:>11.2f}%")
        print(f"{'F1':<20s} {100*s2['f1']:>11.2f}% {100*s3['f1']:>11.2f}%")
        print(f"{'TP':<20s} {s2['tp']:>12d} {s3['tp']:>12d}")
        print(f"{'FP':<20s} {s2['fp']:>12d} {s3['fp']:>12d}")
        print(f"{'FN':<20s} {s2['fn']:>12d} {s3['fn']:>12d}")

        # Stage 1 info
        s1 = r['stage1']
        print(f"\n  Stage 1 (Patchwork++ vanilla):")
        print(f"    Obstaculos como ground: {s1['n_misclassified']} ({s1['misclass_pct']:.2f}%)")
        print(f"    Ground precision: {s1['ground_precision']:.1f}%")
        print(f"    Criticos perdidos: {s1['critical_miss']}")

        # Stage 1 con wall rejection (dentro del pipeline)
        s2_info = r['stage2']
        print(f"  Stage 1 (con wall rejection):")
        print(f"    Obstaculos como ground: {s2_info['stage1_misclass']} ({s2_info['stage1_misclass_pct']:.2f}%)")

    # Guardar resultados en JSON para el .md
    output_file = Path(__file__).parent / "results_both_sequences.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResultados guardados en: {output_file}")

    print(f"\n{'='*80}")
    print("TEST COMPLETADO")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
