#!/usr/bin/env python3
"""
Ablation study de los stages del pipeline.

Evalúa el impacto de cada stage en la detección de obstáculos:
  A) PW++ vanilla (sin WR, sin curb, sin DBSCAN)
  B) PW++ + WR
  C) PW++ + WR + DBSCAN
  D) PW++ + WR + Curb + DBSCAN

Métricas: Precision, Recall, F1, IoU
Protocolo: train (seq 00-07, 09-10) + val (seq 08)

Uso:
    python3 tests/test_ablation_stages.py --stride 5
    python3 tests/test_ablation_stages.py --stride 10 --seq 00 04
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
                        VELODYNE_ROOT, OBSTACLE_LABELS, IGNORE_LABELS)

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


def load_frames(seqs, stride):
    frames = []
    for seq in seqs:
        sids = discover_scan_ids(seq, stride)
        for sid in sids:
            bf = str(get_scan_file(seq, sid))
            lf = str(get_label_file(seq, sid, use_curb=True))
            frames.append((bf, lf))
    return frames


def evaluate(pipe, frames):
    tp, fp, fn = 0, 0, 0

    for bf, lf in frames:
        pts = np.fromfile(bf, dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl = np.fromfile(lf, dtype=np.uint32) & 0xFFFF

        gt = np.isin(lbl, OBSTACLE_LABELS)
        valid = ~np.isin(lbl, IGNORE_LABELS)

        r = pipe.stage3_complete(pts)
        obs = r['obs_mask']

        g = gt & valid
        p = obs & valid
        tp += int(np.sum(g & p))
        fp += int(np.sum(~g & p))
        fn += int(np.sum(g & ~p))

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp / (tp + fp + fn) * 100 if (tp + fp + fn) > 0 else 0
    return prec, rec, f1, iou


def main():
    parser = argparse.ArgumentParser(description='Ablation study de stages del pipeline')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--seq', nargs='*', default=None, help='Secuencias específicas (default: todas)')
    args = parser.parse_args()

    if args.seq:
        train_seqs = [s for s in args.seq if s in SEMANTICKITTI_TRAIN]
        val_seqs = [s for s in args.seq if s in SEMANTICKITTI_VAL]
        if not val_seqs:
            val_seqs = []
    else:
        train_seqs = SEMANTICKITTI_TRAIN
        val_seqs = SEMANTICKITTI_VAL

    print("=" * 90)
    print("ABLATION STUDY — STAGES DEL PIPELINE (detección de obstáculos)")
    print("=" * 90)

    # Cargar datos
    print(f"\nCargando datos (stride={args.stride})...")
    train_frames = load_frames(train_seqs, args.stride)
    val_frames = load_frames(val_seqs, args.stride) if val_seqs else []
    print(f"  Train: {len(train_frames)} frames ({train_seqs})")
    if val_frames:
        print(f"  Val:   {len(val_frames)} frames ({val_seqs})")

    # Configuraciones a evaluar
    configs = [
        ("A) PW++ vanilla",              PipelineConfig(
            enable_hybrid_wall_rejection=False,
            enable_curb_detection=False,
            enable_cluster_filtering=False,
            verbose=False)),
        ("B) PW++ + WR",                 PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=False,
            enable_cluster_filtering=False,
            verbose=False)),
        ("C) PW++ + WR + DBSCAN",        get_kitti_config(curb=False)),
        ("D) PW++ + WR + Curb + DBSCAN", get_kitti_config(curb=True)),
    ]

    # Evaluar
    header = f"  {'Config':40s} | {'P':>7s} {'R':>7s} {'F1':>7s} {'IoU':>7s}"
    sep = f"  {'-'*40}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}"

    for split_name, frames in [("TRAIN", train_frames), ("VAL", val_frames)]:
        if not frames:
            continue
        print(f"\n--- {split_name} ({len(frames)} frames) ---")
        print(header)
        print(sep)

        t0 = time.time()
        for name, cfg in configs:
            cfg.verbose = False
            pipe = LidarPipelineSuite(cfg)
            prec, rec, f1, iou = evaluate(pipe, frames)
            print(f"  {name:40s} | {prec:6.1f}% {rec:6.1f}% {f1:6.1f}% {iou:6.1f}%")
        elapsed = time.time() - t0
        print(f"\n  Tiempo: {elapsed:.0f}s")

    print(f"\n{'='*90}")


if __name__ == '__main__':
    main()
