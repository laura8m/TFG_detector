#!/usr/bin/env python3
"""
data_paths.py
=============
Módulo centralizado para construir rutas a los datos KITTI Odometry.

Estructura esperada:
    data_odometry_velodyne/dataset/sequences/{seq}/velodyne/{scan_id:06d}.bin
    data_odometry_labels/dataset/sequences/{seq}/labels/{scan_id:06d}.label
    data_odometry_labels/dataset/sequences/{seq}/poses.txt

Uso:
    from data_paths import get_scan_file, get_label_file, get_sequence_info

    scan = get_scan_file('04', 0)       # -> Path(...000000.bin)
    info = get_sequence_info('00')      # -> dict con data_dir, label_dir, poses_file
"""

from pathlib import Path

# Directorio base del proyecto (donde reside este fichero)
BASE_DIR = Path(__file__).parent.resolve()

VELODYNE_ROOT = BASE_DIR / "data_odometry_velodyne" / "dataset" / "sequences"
LABELS_ROOT = BASE_DIR / "data_odometry_labels" / "dataset" / "sequences"


def get_velodyne_dir(seq: str) -> Path:
    """Directorio de archivos .bin para una secuencia."""
    return VELODYNE_ROOT / seq / "velodyne"


def get_labels_dir(seq: str) -> Path:
    """Directorio de archivos .label para una secuencia."""
    return LABELS_ROOT / seq / "labels"


def get_poses_file(seq: str) -> Path:
    """Archivo poses.txt de una secuencia."""
    return LABELS_ROOT / seq / "poses.txt"


def get_scan_file(seq: str, scan_id: int) -> Path:
    """Ruta completa a un .bin concreto."""
    return get_velodyne_dir(seq) / f"{scan_id:06d}.bin"


def get_label_file(seq: str, scan_id: int) -> Path:
    """Ruta completa a un .label concreto."""
    return get_labels_dir(seq) / f"{scan_id:06d}.label"


def get_sequence_info(seq: str) -> dict:
    """
    Devuelve dict compatible con el formato SEQUENCES usado en run_pipeline_viz, tests, etc.

    Returns:
        dict con claves 'data_dir', 'label_dir', 'poses_file'
    """
    return {
        'data_dir': VELODYNE_ROOT / seq,
        'label_dir': get_labels_dir(seq),
        'poses_file': str(get_poses_file(seq)),
    }
