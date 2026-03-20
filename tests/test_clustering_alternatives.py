#!/usr/bin/env python3
"""
Test: Alternativas de clustering para Stage 3.

Compara:
1. DBSCAN actual (eps=0.8, min_samples=8, min_pts=30) — baseline
2. HDBSCAN (hierarchical, sin eps fijo)
3. DBSCAN con eps adaptativo por distancia
4. Range-image connected components (sin clustering 3D)

Evaluado en Seq 04 (highway) y Seq 00 (urban), 10 frames cada una.
"""

import sys
import numpy as np
from pathlib import Path
import time
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_scan_file, get_label_file


def load_kitti_scan(scan_id, seq='04'):
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF
    else:
        semantic_labels = np.zeros(len(points), dtype=np.uint32)
    return points, semantic_labels


def get_gt_obstacle_mask(semantic_labels):
    obstacle_labels = [
        10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
        50, 51, 52, 70, 71, 80, 81, 99,
        252, 253, 254, 255, 256, 257, 258, 259
    ]
    mask = np.zeros(len(semantic_labels), dtype=bool)
    for label in obstacle_labels:
        mask |= (semantic_labels == label)
    return mask


def compute_metrics(gt_mask, pred_mask):
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


# ========================================
# ALTERNATIVA 1: HDBSCAN
# ========================================

def cluster_hdbscan(obs_pts, min_cluster_size, min_samples):
    """HDBSCAN: clustering jerárquico sin eps fijo."""
    try:
        from sklearn.cluster import HDBSCAN as HDBSCAN_sklearn
        hdb = HDBSCAN_sklearn(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_jobs=-1
        )
        return hdb.fit_predict(obs_pts)
    except ImportError:
        # Fallback: hdbscan package
        import hdbscan
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1
        )
        return hdb.fit_predict(obs_pts)


# ========================================
# ALTERNATIVA 2: DBSCAN eps adaptativo
# ========================================

def cluster_adaptive_dbscan(obs_pts, eps_base, min_samples, r_ref=10.0, eps_max=3.0):
    """
    DBSCAN con eps adaptativo por distancia.
    Divide puntos en bandas de distancia, aplica DBSCAN con eps escalado.
    """
    ranges = np.sqrt(obs_pts[:, 0]**2 + obs_pts[:, 1]**2)

    # Bandas de distancia
    bands = [(0, 15), (15, 30), (30, 50), (50, 80)]
    all_labels = np.full(len(obs_pts), -1, dtype=np.int32)
    label_offset = 0

    for r_min, r_max in bands:
        band_mask = (ranges >= r_min) & (ranges < r_max)
        if band_mask.sum() < min_samples:
            continue

        # eps escalado con distancia media de la banda
        r_mid = (r_min + r_max) / 2.0
        eps_scaled = min(eps_base * (r_mid / r_ref), eps_max)

        band_pts = obs_pts[band_mask]
        db = DBSCAN(eps=eps_scaled, min_samples=min_samples, n_jobs=-1)
        band_labels = db.fit_predict(band_pts)

        # Reasignar labels con offset para no colisionar entre bandas
        valid = band_labels >= 0
        band_labels[valid] += label_offset
        label_offset += (band_labels.max() + 1) if valid.any() else 0

        all_labels[band_mask] = band_labels

    return all_labels


# ========================================
# ALTERNATIVA 3: Range-image connected components
# ========================================

def cluster_range_image_cc(points, obs_mask, n_rings=64, n_cols=2048, alpha=0.1):
    """
    Connected components en range image con discontinuidad de profundidad.
    Dos pixels adyacentes son conectados si |r1-r2| < alpha * min(r1,r2).
    """
    # Proyectar a range image
    ranges = np.sqrt(np.sum(points**2, axis=1))
    xy_range = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

    # Ángulos
    azimuth = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
    elevation = np.arctan2(points[:, 2], xy_range)  # vertical

    # Mapear a pixels
    col = ((azimuth + np.pi) / (2 * np.pi) * n_cols).astype(np.int32) % n_cols

    # Para HDL-64E: FOV -24.8° a +2°
    fov_up = 2.0 * np.pi / 180
    fov_down = -24.8 * np.pi / 180
    fov_total = fov_up - fov_down
    row = ((fov_up - elevation) / fov_total * n_rings).astype(np.int32)
    row = np.clip(row, 0, n_rings - 1)

    # Crear range image con índice del punto
    range_img = np.full((n_rings, n_cols), -1, dtype=np.int32)
    range_vals = np.full((n_rings, n_cols), 0.0, dtype=np.float32)
    obs_img = np.zeros((n_rings, n_cols), dtype=bool)

    # Llenar (último punto gana en caso de colisión)
    for i in range(len(points)):
        r, c = row[i], col[i]
        range_img[r, c] = i
        range_vals[r, c] = ranges[i]
        if obs_mask[i]:
            obs_img[r, c] = True

    # Connected components con BFS
    labels_img = np.full((n_rings, n_cols), -1, dtype=np.int32)
    current_label = 0

    for r in range(n_rings):
        for c in range(n_cols):
            if not obs_img[r, c] or labels_img[r, c] >= 0:
                continue

            # BFS
            queue = [(r, c)]
            labels_img[r, c] = current_label
            head = 0
            while head < len(queue):
                cr, cc = queue[head]
                head += 1
                rv = range_vals[cr, cc]

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, (cc + dc) % n_cols
                    if 0 <= nr < n_rings and obs_img[nr, nc] and labels_img[nr, nc] < 0:
                        nv = range_vals[nr, nc]
                        # Discontinuidad de profundidad
                        if abs(rv - nv) < alpha * min(rv, nv):
                            labels_img[nr, nc] = current_label
                            queue.append((nr, nc))

            current_label += 1

    # Mapear labels de image a puntos
    point_labels = np.full(len(points), -1, dtype=np.int32)
    for r in range(n_rings):
        for c in range(n_cols):
            idx = range_img[r, c]
            if idx >= 0 and labels_img[r, c] >= 0:
                point_labels[idx] = labels_img[r, c]

    return point_labels


# ========================================
# APLICAR CLUSTERING Y FILTRAR
# ========================================

def apply_custom_clustering(points, stage2_result, method, min_pts=30, **kwargs):
    """Aplica un método de clustering alternativo y filtra clusters pequeños."""
    obs_mask = stage2_result['obs_mask'].copy()
    obs_indices = np.where(obs_mask)[0]
    N = len(points)

    if len(obs_indices) == 0:
        return obs_mask

    obs_pts = points[obs_indices]

    # Voxel downsampling (igual que pipeline actual)
    voxel_size = 0.24
    vox_coords = np.floor(obs_pts / voxel_size).astype(np.int32)
    vox_keys = (vox_coords[:, 0].astype(np.int64) * 1000003 +
                vox_coords[:, 1].astype(np.int64) * 1009 +
                vox_coords[:, 2].astype(np.int64))
    unique_keys, inverse, counts = np.unique(vox_keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)
    voxel_centroids = np.zeros((n_voxels, 3), dtype=np.float64)
    for dim in range(3):
        np.add.at(voxel_centroids[:, dim], inverse, obs_pts[:, dim])
    voxel_centroids /= counts[:, np.newaxis]
    voxel_centroids = voxel_centroids.astype(np.float32)

    t0 = time.time()

    if method == 'hdbscan':
        voxel_labels = cluster_hdbscan(voxel_centroids, **kwargs)
    elif method == 'adaptive_dbscan':
        voxel_labels = cluster_adaptive_dbscan(voxel_centroids, **kwargs)
    elif method == 'range_cc':
        # Range CC trabaja directamente sobre todos los puntos (sin voxel downsampling)
        point_labels = cluster_range_image_cc(points, obs_mask, **kwargs)
        # Filtrar clusters pequeños
        cluster_labels_obs = point_labels[obs_indices]
        unique_labels = np.unique(cluster_labels_obs)
        valid_clusters = set()
        for label in unique_labels:
            if label == -1:
                continue
            if (cluster_labels_obs == label).sum() >= min_pts:
                valid_clusters.add(label)
        if valid_clusters:
            valid_mask = np.isin(cluster_labels_obs, np.array(list(valid_clusters)))
        else:
            valid_mask = np.zeros(len(obs_indices), dtype=bool)
        obs_mask_new = np.zeros(N, dtype=bool)
        obs_mask_new[obs_indices[valid_mask]] = True
        dt = (time.time() - t0) * 1000
        return obs_mask_new, dt
    else:
        raise ValueError(f"Unknown method: {method}")

    dt = (time.time() - t0) * 1000

    # Propagar labels de voxel a puntos
    cluster_labels_obs = voxel_labels[inverse]

    # Filtrar clusters pequeños
    unique_labels = np.unique(cluster_labels_obs)
    valid_clusters = set()
    for label in unique_labels:
        if label == -1:
            continue
        if (cluster_labels_obs == label).sum() >= min_pts:
            valid_clusters.add(label)

    if valid_clusters:
        valid_mask = np.isin(cluster_labels_obs, np.array(list(valid_clusters)))
    else:
        valid_mask = np.zeros(len(obs_indices), dtype=bool)

    obs_mask_new = np.zeros(N, dtype=bool)
    obs_mask_new[obs_indices[valid_mask]] = True
    return obs_mask_new, dt


# ========================================
# MAIN
# ========================================

def run_test(seqs, n_frames, scan_start):
    # Pre-cargar datos con Stage 2
    print("Cargando datos y ejecutando Stage 2...")
    pipeline_base = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=False,
        verbose=False
    ))

    data = {}
    for seq in seqs:
        for i in range(n_frames):
            scan_id = scan_start + i
            pts, labels = load_kitti_scan(scan_id, seq)
            gt_mask = get_gt_obstacle_mask(labels)
            r_s2 = pipeline_base.stage2_complete(pts)
            data[(seq, scan_id)] = (pts, gt_mask, r_s2)

    # Definir métodos a probar
    methods = [
        ('DBSCAN actual (baseline)', 'pipeline', {}),
        ('HDBSCAN mcs=10 ms=5', 'hdbscan', {'min_cluster_size': 10, 'min_samples': 5}),
        ('HDBSCAN mcs=15 ms=5', 'hdbscan', {'min_cluster_size': 15, 'min_samples': 5}),
        ('HDBSCAN mcs=20 ms=3', 'hdbscan', {'min_cluster_size': 20, 'min_samples': 3}),
        ('HDBSCAN mcs=20 ms=5', 'hdbscan', {'min_cluster_size': 20, 'min_samples': 5}),
        ('HDBSCAN mcs=30 ms=5', 'hdbscan', {'min_cluster_size': 30, 'min_samples': 5}),
        ('HDBSCAN mcs=30 ms=8', 'hdbscan', {'min_cluster_size': 30, 'min_samples': 8}),
        ('AdaptDBSCAN eps=0.5 rref=10', 'adaptive_dbscan', {'eps_base': 0.5, 'min_samples': 4, 'r_ref': 10.0}),
        ('AdaptDBSCAN eps=0.8 rref=10', 'adaptive_dbscan', {'eps_base': 0.8, 'min_samples': 4, 'r_ref': 10.0}),
        ('AdaptDBSCAN eps=0.5 rref=15', 'adaptive_dbscan', {'eps_base': 0.5, 'min_samples': 4, 'r_ref': 15.0}),
        ('AdaptDBSCAN eps=0.8 rref=15', 'adaptive_dbscan', {'eps_base': 0.8, 'min_samples': 4, 'r_ref': 15.0}),
        ('Range CC alpha=0.05', 'range_cc', {'alpha': 0.05}),
        ('Range CC alpha=0.10', 'range_cc', {'alpha': 0.10}),
        ('Range CC alpha=0.15', 'range_cc', {'alpha': 0.15}),
    ]

    pipeline_dbscan = LidarPipelineSuite(PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_cluster_filtering=True,
        verbose=False
    ))

    results = []

    for name, method, kwargs in methods:
        print(f"\n  Probando: {name}...")
        combo_results = {}
        total_time = 0
        n_evals = 0

        for seq in seqs:
            all_tp, all_fp, all_fn = 0, 0, 0
            for i in range(n_frames):
                scan_id = scan_start + i
                pts, gt_mask, r_s2 = data[(seq, scan_id)]

                if method == 'pipeline':
                    # Baseline: usar pipeline completo
                    r_s3 = pipeline_dbscan.stage3_cluster_filtering(pts, r_s2)
                    pred_mask = r_s3['obs_mask']
                    dt = r_s3['timing_stage3_ms']
                else:
                    pred_mask, dt = apply_custom_clustering(
                        pts, r_s2, method, min_pts=30, **kwargs
                    )

                total_time += dt
                n_evals += 1
                m = compute_metrics(gt_mask, pred_mask)
                all_tp += m['tp']
                all_fp += m['fp']
                all_fn += m['fn']

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            combo_results[seq] = {'precision': p, 'recall': r, 'f1': f1}

        mean_f1 = np.mean([combo_results[s]['f1'] for s in seqs])
        mean_p = np.mean([combo_results[s]['precision'] for s in seqs])
        mean_r = np.mean([combo_results[s]['recall'] for s in seqs])
        avg_time = total_time / n_evals if n_evals > 0 else 0

        results.append({
            'name': name, 'per_seq': combo_results,
            'mean_f1': mean_f1, 'mean_p': mean_p, 'mean_r': mean_r,
            'avg_ms': avg_time,
        })
        print(f"    -> Mean F1={100*mean_f1:.2f}% P={100*mean_p:.1f}% R={100*mean_r:.1f}% ({avg_time:.0f}ms)")

    # Tabla final
    results.sort(key=lambda x: x['mean_f1'], reverse=True)

    print("\n" + "=" * 120)
    print("RESULTADOS CLUSTERING (ordenados por Mean F1)")
    print("=" * 120)
    print(f"{'Rank':<5} {'Método':<30} | {'04 P%':>6} {'04 R%':>6} {'04 F1%':>7} | {'00 P%':>6} {'00 R%':>6} {'00 F1%':>7} | {'Mean F1':>8} {'ms':>6}")
    print("-" * 120)

    baseline_f1 = None
    for i, r in enumerate(results):
        if r['name'] == 'DBSCAN actual (baseline)':
            baseline_f1 = r['mean_f1']
        s04 = r['per_seq'].get('04', {'precision': 0, 'recall': 0, 'f1': 0})
        s00 = r['per_seq'].get('00', {'precision': 0, 'recall': 0, 'f1': 0})
        delta = f" Δ={100*(r['mean_f1']-baseline_f1):+.2f}%" if baseline_f1 else ""
        print(f"{i+1:<5} {r['name']:<30} | {100*s04['precision']:>6.1f} {100*s04['recall']:>6.1f} {100*s04['f1']:>7.1f} | {100*s00['precision']:>6.1f} {100*s00['recall']:>6.1f} {100*s00['f1']:>7.1f} | {100*r['mean_f1']:>7.2f}% {r['avg_ms']:>5.0f}ms{delta}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test clustering alternatives')
    parser.add_argument('--seq', type=str, default='both', choices=['00', '04', 'both'])
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--scan_start', type=int, default=0)
    args = parser.parse_args()

    seqs = ['04', '00'] if args.seq == 'both' else [args.seq]
    run_test(seqs, args.n_frames, args.scan_start)
