#!/usr/bin/env python3
"""
Ablation study de detección de bordillos.

Evalúa el impacto de activar curb detection en:
  - CurbRecall: % de bordillos GT detectados como obstáculo
  - CurbPrecision: % de detecciones curb que son bordillos reales
  - F1 global: cómo afecta al F1 de obstáculos
  - IoU global: cómo afecta al IoU de obstáculos

Compara:
  1) Sin curb detection (baseline)
  2) Con curb detection (gap=1)
  3) Con curb detection (gap=2)
  4) Con curb detection (gap=3, default)
  5) Con curb detection (gap=4)

Protocolo: val (seq 08) con labels 3D-Curb

Uso:
    python3 tests/test_ablation_curb.py --stride 5
    python3 tests/test_ablation_curb.py --stride 10 --seq 00 04
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from pipeline_params import get_kitti_config
from data_paths import (get_scan_file, get_label_file, get_velodyne_dir, get_labels_dir,
                        VELODYNE_ROOT, OBSTACLE_LABELS, IGNORE_LABELS, CURB_LABEL)

SEMANTICKITTI_TRAIN = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
SEMANTICKITTI_VAL = ['08']


def discover_scan_ids(seq, stride=5):
    vel_dir = get_velodyne_dir(seq)
    lab_dir = get_labels_dir(seq, use_curb=True)
    if not vel_dir.exists() or not lab_dir.exists():
        return []
    vel_ids = {int(f.stem) for f in vel_dir.glob('*.bin')}
    lab_ids = {int(f.stem) for f in lab_dir.glob('*.label')}
    return sorted(vel_ids & lab_ids)[::stride]


def precompute_stage1(seqs, stride):
    """Cachea Stage 1 (PW++ + WR) para no repetirlo por cada config de curb."""
    pipe = LidarPipelineSuite(get_kitti_config(curb=False))

    cached = []
    for seq in seqs:
        for sid in discover_scan_ids(seq, stride):
            pts = np.fromfile(str(get_scan_file(seq, sid)), dtype=np.float32).reshape(-1, 4)[:, :3]
            lbl = np.fromfile(str(get_label_file(seq, sid, use_curb=True)), dtype=np.uint32) & 0xFFFF

            s1 = pipe.stage1_complete(pts)
            N = len(pts)
            base_obs = np.zeros(N, dtype=bool)
            base_obs[s1['nonground_indices']] = True

            cached.append({
                'pts': pts,
                'lbl': lbl,
                'gt_obs': np.isin(lbl, OBSTACLE_LABELS),
                'gt_curb': lbl == CURB_LABEL,
                'valid': ~np.isin(lbl, IGNORE_LABELS),
                'base_obs': base_obs,
                'ground_mask': ~base_obs,
            })
    return cached, pipe


def evaluate_curb(pipe, cached, enable_curb, ring_gap=3):
    """Evalúa F1/IoU global + CurbRecall/CurbPrecision."""
    if enable_curb:
        pipe.config.enable_curb_detection = True
        pipe.config.curb_ring_gap = ring_gap

    tp, fp, fn = 0, 0, 0
    curb_tp, curb_total = 0, 0
    curb_det_tp, curb_det_total = 0, 0  # precision de curb_mask

    for d in cached:
        if enable_curb:
            curb_mask = pipe.detect_curbs(d['pts'], d['ground_mask'])
            obs_mask = d['base_obs'] | curb_mask
        else:
            curb_mask = np.zeros(len(d['pts']), dtype=bool)
            obs_mask = d['base_obs']

        g = d['gt_obs'] & d['valid']
        p = obs_mask & d['valid']
        tp += int(np.sum(g & p))
        fp += int(np.sum(~g & p))
        fn += int(np.sum(g & ~p))

        # CurbRecall: de los curb GT, cuántos detectamos como obs
        nc = int(d['gt_curb'].sum())
        if nc > 0:
            curb_tp += int(np.sum(d['gt_curb'] & obs_mask))
            curb_total += nc

        # CurbPrecision: de los puntos marcados como curb, cuántos son curb GT
        n_det = int(curb_mask.sum())
        if n_det > 0:
            curb_det_tp += int(np.sum(curb_mask & d['gt_curb']))
            curb_det_total += n_det

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp / (tp + fp + fn) * 100 if (tp + fp + fn) > 0 else 0
    curb_recall = curb_tp / curb_total * 100 if curb_total > 0 else 0
    curb_prec = curb_det_tp / curb_det_total * 100 if curb_det_total > 0 else 0

    return {
        'prec': prec, 'rec': rec, 'f1': f1, 'iou': iou,
        'curb_recall': curb_recall, 'curb_prec': curb_prec,
        'curb_tp': curb_tp, 'curb_total': curb_total,
        'curb_det_total': curb_det_total,
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation study de curb detection')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--seq', nargs='*', default=None)
    args = parser.parse_args()

    if args.seq:
        train_seqs = [s for s in args.seq if s in SEMANTICKITTI_TRAIN]
        val_seqs = [s for s in args.seq if s in SEMANTICKITTI_VAL]
        if not val_seqs:
            val_seqs = train_seqs  # usar como val si no hay seq 08
    else:
        train_seqs = SEMANTICKITTI_TRAIN
        val_seqs = SEMANTICKITTI_VAL

    print("=" * 100)
    print("ABLATION STUDY — DETECCIÓN DE BORDILLOS")
    print("=" * 100)

    for split_name, seqs in [("TRAIN", train_seqs), ("VAL", val_seqs)]:
        print(f"\nPrecomputando Stage 1 para {split_name}...")
        t0 = time.time()
        cached, pipe = precompute_stage1(seqs, args.stride)
        print(f"  {len(cached)} frames en {time.time()-t0:.1f}s")

        if not cached:
            print(f"  Sin datos para {split_name}, saltando")
            continue

        # Baseline
        bl = evaluate_curb(pipe, cached, enable_curb=False)

        # Con curb en distintos gaps
        results = []
        for gap in [1, 2, 3, 4]:
            r = evaluate_curb(pipe, cached, enable_curb=True, ring_gap=gap)
            results.append((gap, r))

        # Mostrar resultados
        print(f"\n--- {split_name} ({len(cached)} frames, {seqs}) ---")
        print()
        print(f"  {'Config':30s} | {'F1':>7s} {'ΔF1':>7s} {'IoU':>7s} {'ΔIoU':>7s} | {'CurbR':>7s} {'CurbP':>7s} {'Curb det':>9s}")
        print(f"  {'-'*30}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*7}-{'-'*7}-{'-'*9}")

        print(f"  {'Baseline (sin curb)':30s} | {bl['f1']:6.1f}% {'—':>7s} {bl['iou']:6.1f}% {'—':>7s} | {bl['curb_recall']:6.1f}% {'—':>7s} {'—':>9s}")

        for gap, r in results:
            df1 = r['f1'] - bl['f1']
            diou = r['iou'] - bl['iou']
            print(f"  {'Con curb (gap=' + str(gap) + ')':30s} | {r['f1']:6.1f}% {df1:+6.1f}% {r['iou']:6.1f}% {diou:+6.1f}% | {r['curb_recall']:6.1f}% {r['curb_prec']:6.1f}% {r['curb_det_total']:>9d}")

        # Resumen
        best = max(results, key=lambda x: x[1]['curb_recall'])
        best_f1 = max(results, key=lambda x: x[1]['f1'])
        print(f"\n  Mejor CurbRecall: gap={best[0]} → CurbR={best[1]['curb_recall']:.1f}%, F1={best[1]['f1']:.1f}% (ΔF1={best[1]['f1']-bl['f1']:+.1f}%)")
        print(f"  Mejor F1:         gap={best_f1[0]} → F1={best_f1[1]['f1']:.1f}% (ΔF1={best_f1[1]['f1']-bl['f1']:+.1f}%), CurbR={best_f1[1]['curb_recall']:.1f}%")

    print(f"\n{'='*100}")


if __name__ == '__main__':
    main()
