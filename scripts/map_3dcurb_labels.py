#!/usr/bin/env python3
"""
Mapea labels curb=3 de 3D-Curb al point cloud completo de SemanticKITTI.

Genera nuevos archivos .label con las 28 clases originales + curb=3.

Uso en mazinger:
    python3 map_3dcurb_labels.py

Rutas configurables abajo en BASE_PATHS.
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import sys
import time

# =============================================================================
# CONFIGURACIÓN DE RUTAS (ajustar según máquina)
# =============================================================================
BASE = Path("/home/gsanchez/laura/sota_ws/TFG_detector")

VELODYNE_DIR = BASE / "data_odometry_velodyne" / "dataset" / "sequences"
LABELS_DIR = BASE / "data_odometry_labels" / "dataset" / "sequences"
CURB3D_DIR = BASE / "3d_curb_labels"
OUTPUT_DIR = BASE / "data_curb_mapped_labels" / "dataset" / "sequences"

SEQUENCES = [f"{i:02d}" for i in range(11)]  # 00-10
TOLERANCE = 0.01  # 1cm — match exacto
CURB_LABEL = 3


def map_frame(velodyne_bin, semantickitti_label, curb3d_bin, curb3d_label):
    """
    Transfiere labels curb=3 de 3D-Curb al point cloud completo de SemanticKITTI.

    Returns:
        new_labels: array uint32 con labels actualizadas
        n_curb_mapped: número de puntos curb mapeados
        n_curb_total: número total de puntos curb en 3D-Curb
    """
    # Cargar point cloud completo
    full_pts = np.fromfile(velodyne_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
    full_labels = np.fromfile(semantickitti_label, dtype=np.uint32).copy()

    # Cargar 3D-Curb (recortado)
    curb_pts = np.fromfile(curb3d_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
    curb_labels = np.fromfile(curb3d_label, dtype=np.uint32)

    # Filtrar solo puntos con label curb=3
    curb_semantic = curb_labels & 0xFFFF
    curb_mask = curb_semantic == CURB_LABEL
    n_curb_total = curb_mask.sum()

    if n_curb_total == 0:
        return full_labels, 0, 0

    curb_points = curb_pts[curb_mask]

    # Buscar correspondencia en el point cloud completo
    tree = cKDTree(full_pts)
    dist, idx = tree.query(curb_points, k=1)

    # Transferir label curb=3 (mantener instance ID en upper 16 bits)
    valid = dist < TOLERANCE
    n_curb_mapped = valid.sum()
    full_labels[idx[valid]] = (full_labels[idx[valid]] & 0xFFFF0000) | CURB_LABEL

    return full_labels, n_curb_mapped, n_curb_total


def process_sequence(seq):
    """Procesa todos los frames de una secuencia."""
    vel_dir = VELODYNE_DIR / seq / "velodyne"
    lab_dir = LABELS_DIR / seq / "labels"
    curb_vel_dir = CURB3D_DIR / seq / "velodyne"
    curb_lab_dir = CURB3D_DIR / seq / "labels"
    out_dir = OUTPUT_DIR / seq / "labels"

    # Verificar que existen los datos
    if not curb_vel_dir.exists():
        print(f"  [SKIP] Seq {seq}: no existe {curb_vel_dir}")
        return 0, 0, 0
    if not vel_dir.exists():
        print(f"  [SKIP] Seq {seq}: no existe {vel_dir}")
        return 0, 0, 0
    if not lab_dir.exists():
        print(f"  [SKIP] Seq {seq}: no existe {lab_dir}")
        return 0, 0, 0

    # Crear directorio de salida
    out_dir.mkdir(parents=True, exist_ok=True)

    # Listar frames disponibles en 3D-Curb
    curb_bins = sorted(curb_vel_dir.glob("*.bin"))
    total_curb = 0
    total_mapped = 0
    n_frames = 0

    for curb_bin in curb_bins:
        frame_id = curb_bin.stem  # e.g. "000042"

        vel_file = vel_dir / f"{frame_id}.bin"
        lab_file = lab_dir / f"{frame_id}.label"
        curb_lab_file = curb_lab_dir / f"{frame_id}.label"
        out_file = out_dir / f"{frame_id}.label"

        if not vel_file.exists() or not lab_file.exists():
            continue

        new_labels, n_mapped, n_total = map_frame(vel_file, lab_file, curb_bin, curb_lab_file)
        new_labels.tofile(str(out_file))

        total_curb += n_total
        total_mapped += n_mapped
        n_frames += 1

    return n_frames, total_mapped, total_curb


def main():
    print("=" * 60)
    print("Mapeo de labels 3D-Curb → SemanticKITTI completo")
    print("=" * 60)
    print(f"Velodyne:  {VELODYNE_DIR}")
    print(f"Labels:    {LABELS_DIR}")
    print(f"3D-Curb:   {CURB3D_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print()

    t0 = time.time()
    grand_frames = 0
    grand_mapped = 0
    grand_curb = 0

    for seq in SEQUENCES:
        print(f"Procesando seq {seq}...")
        n_frames, n_mapped, n_curb = process_sequence(seq)
        grand_frames += n_frames
        grand_mapped += n_mapped
        grand_curb += n_curb

        if n_frames > 0:
            pct = n_mapped / n_curb * 100 if n_curb > 0 else 0
            print(f"  {n_frames} frames | {n_curb} curb pts | {n_mapped} mapeados ({pct:.1f}%)")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"TOTAL: {grand_frames} frames | {grand_curb} curb pts | {grand_mapped} mapeados")
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
