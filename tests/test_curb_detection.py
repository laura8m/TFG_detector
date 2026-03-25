#!/usr/bin/env python3
"""
Test de detección de bordillos usando 3D-Curb dataset.

Evalúa la Feature 1 (Inter-Ring Height Discontinuity) contra
el GT de bordillos (label 3) del dataset 3D-Curb (basado en SemanticKITTI).

Compara:
  - PW++ vanilla (sin detección de bordillos)
  - PW++ + WR (sin detección de bordillos)
  - PW++ + WR + Feature 1 (con detección de bordillos)
  - PW++ + WR + Feature 1 + delta-r conservador

Métricas: recall de bordillos (¿cuántos bordillos del GT detecta como obstáculo?)
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# Label 3 = curb en 3D-Curb dataset
CURB_LABEL = 3


def main():
    parser = argparse.ArgumentParser(description='Test detección de bordillos con 3D-Curb')
    parser.add_argument('--seq', default='08', help='Secuencia (default: 08)')
    parser.add_argument('--stride', type=int, default=1, help='Stride entre frames')
    parser.add_argument('--curb_labels_dir', default=None, help='Directorio de labels 3D-Curb')
    parser.add_argument('--velodyne_dir', default=None, help='Directorio de velodyne')
    parser.add_argument('--curb_min', type=float, default=0.08, help='Altura mínima bordillo (m)')
    parser.add_argument('--curb_max', type=float, default=0.25, help='Altura máxima bordillo (m)')
    args = parser.parse_args()

    # Auto-detectar paths
    base = Path(__file__).parent.parent

    if args.curb_labels_dir:
        curb_labels_dir = Path(args.curb_labels_dir)
    else:
        # Buscar en ubicaciones comunes
        candidates = [
            base / '3d_curb_labels' / args.seq / 'labels',
            base / 'data_3d_curb' / args.seq / 'labels',
            Path.home() / 'laura' / 'sota_ws' / 'TFG_detector' / '3d_curb_labels' / args.seq / 'labels',
        ]
        curb_labels_dir = None
        for c in candidates:
            if c.exists():
                curb_labels_dir = c
                break
        if curb_labels_dir is None:
            print("ERROR: No se encontró directorio de labels 3D-Curb")
            print("  Usa --curb_labels_dir para especificarlo")
            sys.exit(1)

    if args.velodyne_dir:
        velodyne_dir = Path(args.velodyne_dir)
    else:
        candidates = [
            base / 'data_odometry_velodyne' / 'dataset' / 'sequences' / args.seq / 'velodyne',
            Path.home() / 'laura' / 'sota_ws' / 'TFG_detector' / 'data_odometry_velodyne' / 'dataset' / 'sequences' / args.seq / 'velodyne',
        ]
        velodyne_dir = None
        for c in candidates:
            if c.exists():
                velodyne_dir = c
                break
        if velodyne_dir is None:
            print("ERROR: No se encontró directorio de velodyne")
            sys.exit(1)

    # Obtener frames disponibles en 3D-Curb
    curb_label_files = sorted(glob.glob(str(curb_labels_dir / '*.label')))
    scan_ids = [int(Path(f).stem) for f in curb_label_files]

    # Aplicar stride
    scan_ids = scan_ids[::args.stride]

    print("=" * 80)
    print(f"TEST DETECCIÓN DE BORDILLOS — 3D-Curb Dataset")
    print(f"=" * 80)
    print(f"  Seq: {args.seq} | Frames: {len(scan_ids)} (stride={args.stride})")
    print(f"  Labels: {curb_labels_dir}")
    print(f"  Velodyne: {velodyne_dir}")
    print(f"  Curb height: [{args.curb_min}, {args.curb_max}]m")
    print()

    # Configuraciones a evaluar
    configs = {
        'PW++ vanilla': PipelineConfig(
            enable_hybrid_wall_rejection=False,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False
        ),
        'PW++ + WR': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=False,
            enable_delta_r=False,
            verbose=False
        ),
        'PW++ + WR + Curb': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=True,
            enable_delta_r=False,
            curb_height_min=args.curb_min,
            curb_height_max=args.curb_max,
            verbose=False
        ),
        'PW++ + WR + Curb + delta-r': PipelineConfig(
            enable_hybrid_wall_rejection=True,
            enable_curb_detection=True,
            enable_delta_r=True,
            curb_height_min=args.curb_min,
            curb_height_max=args.curb_max,
            verbose=False
        ),
    }

    results = {}

    for name, config in configs.items():
        print(f"Evaluando: {name}...")
        pipe = LidarPipelineSuite(config)

        total_curb_pts = 0
        detected_curb_pts = 0
        total_curb_as_curb = 0  # Solo para configs con curb_mask
        total_time = 0
        n_frames = 0

        for scan_id in scan_ids:
            # Cargar puntos
            bin_file = velodyne_dir / f'{scan_id:06d}.bin'
            if not bin_file.exists():
                continue
            pts = np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4)[:, :3]

            # Cargar labels 3D-Curb
            lbl_file = curb_labels_dir / f'{scan_id:06d}.label'
            if not lbl_file.exists():
                continue
            labels = np.fromfile(str(lbl_file), dtype=np.uint32) & 0xFFFF

            # GT: puntos que son curb
            curb_gt = labels == CURB_LABEL
            n_curb = int(curb_gt.sum())

            if n_curb == 0:
                continue

            # Ejecutar pipeline
            t0 = time.time()
            result = pipe.stage2_complete(pts)
            t1 = time.time()
            total_time += (t1 - t0)

            # ¿Cuántos bordillos del GT detecta como obstáculo?
            obs_mask = result['obs_mask']
            detected = int(np.sum(curb_gt & obs_mask))

            total_curb_pts += n_curb
            detected_curb_pts += detected

            # Si tiene curb_mask, contar también detecciones específicas de curb
            if 'curb_mask' in result:
                curb_detected = int(np.sum(curb_gt & result['curb_mask']))
                total_curb_as_curb += curb_detected

            n_frames += 1

        recall = detected_curb_pts / total_curb_pts * 100 if total_curb_pts > 0 else 0
        ms_per_frame = total_time / n_frames * 1000 if n_frames > 0 else 0

        results[name] = {
            'total_curb': total_curb_pts,
            'detected': detected_curb_pts,
            'recall': recall,
            'curb_specific': total_curb_as_curb,
            'ms_per_frame': ms_per_frame,
            'n_frames': n_frames,
        }

        curb_specific_str = ""
        if total_curb_as_curb > 0:
            curb_recall = total_curb_as_curb / total_curb_pts * 100
            curb_specific_str = f"  Curb-specific recall: {curb_recall:.1f}%"

        print(f"  {n_frames} frames | Curb recall: {recall:.1f}% "
              f"({detected_curb_pts}/{total_curb_pts}) | {ms_per_frame:.1f} ms/frame"
              f"{curb_specific_str}")
        print()

    # Resumen
    print("=" * 80)
    print("RESUMEN — Detección de bordillos (3D-Curb, seq {})".format(args.seq))
    print("=" * 80)
    print(f"  {'Config':<35} {'Curb Recall':>12} {'Detectados':>12} {'ms/frame':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<35} {r['recall']:>11.1f}% {r['detected']:>8}/{r['total_curb']:<8} {r['ms_per_frame']:>9.1f}")

    # Mejora de Feature 1
    if 'PW++ + WR' in results and 'PW++ + WR + Curb' in results:
        delta = results['PW++ + WR + Curb']['recall'] - results['PW++ + WR']['recall']
        print(f"\n  Mejora Feature 1 sobre WR: +{delta:.1f}% curb recall")

    print()


if __name__ == '__main__':
    main()
