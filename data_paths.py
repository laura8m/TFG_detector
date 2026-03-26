#!/usr/bin/env python3
"""
data_paths.py
=============
Módulo centralizado para rutas y etiquetas de datos KITTI Odometry / 3D-Curb.

Estructura esperada:
    data_odometry_velodyne/dataset/sequences/{seq}/velodyne/{scan_id:06d}.bin
    data_odometry_labels/dataset/sequences/{seq}/labels/{scan_id:06d}.label
    data_curb_mapped_labels/dataset/sequences/{seq}/labels/{scan_id:06d}.label

Uso:
    from data_paths import (get_scan_file, get_label_file, get_sequence_info,
                            OBSTACLE_LABELS, GROUND_LABELS, IGNORE_LABELS)

    # Labels con bordillos (3D-Curb mapeados):
    label = get_label_file('00', 42, use_curb=True)
"""

import numpy as np
from pathlib import Path

# ================================================================================
# RUTAS
# ================================================================================

BASE_DIR = Path(__file__).parent.resolve()

VELODYNE_ROOT = BASE_DIR / "data_odometry_velodyne" / "dataset" / "sequences"
LABELS_ROOT = BASE_DIR / "data_odometry_labels" / "dataset" / "sequences"
CURB_LABELS_ROOT = BASE_DIR / "data_curb_mapped_labels" / "dataset" / "sequences"

# ================================================================================
# ETIQUETAS SEMANTICKITTI (centralizadas)
# ================================================================================

# Obstáculos: vehículos, personas, estructuras, vegetación, postes, objetos móviles, bordillos
OBSTACLE_LABELS = np.array([
    3,                                      # curb (3D-Curb dataset)
    10, 11, 13, 15, 16, 18, 20,            # vehículos
    30, 31, 32,                             # personas
    50, 51,                                 # estructuras (fence, other-structure)
    70, 71,                                 # vegetación
    80, 81,                                 # postes
    252, 253, 254, 255, 256, 257, 258, 259  # objetos móviles
], dtype=np.uint32)

# Suelo: road, parking, sidewalk, other-ground, lane-marking, terrain
GROUND_LABELS = np.array([40, 44, 48, 49, 60, 72], dtype=np.uint32)

# Ignorados en evaluación (unlabeled, outlier, other-structure, other-object)
IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)

# Label de bordillo (3D-Curb)
CURB_LABEL = np.uint32(3)


# ================================================================================
# FUNCIONES DE RUTAS
# ================================================================================

def get_velodyne_dir(seq: str) -> Path:
    """Directorio de archivos .bin para una secuencia."""
    return VELODYNE_ROOT / seq / "velodyne"


def get_labels_dir(seq: str, use_curb: bool = True) -> Path:
    """Directorio de archivos .label para una secuencia.
    Por defecto usa labels con bordillos (3D-Curb) si existen."""
    if use_curb:
        curb_dir = CURB_LABELS_ROOT / seq / "labels"
        if curb_dir.exists():
            return curb_dir
    return LABELS_ROOT / seq / "labels"


def get_poses_file(seq: str) -> Path:
    """Archivo poses.txt de una secuencia."""
    return LABELS_ROOT / seq / "poses.txt"


def get_scan_file(seq: str, scan_id: int) -> Path:
    """Ruta completa a un .bin concreto."""
    return get_velodyne_dir(seq) / f"{scan_id:06d}.bin"


def get_label_file(seq: str, scan_id: int, use_curb: bool = True) -> Path:
    """
    Ruta completa a un .label concreto.

    Por defecto usa labels con bordillos (3D-Curb) si existen.
    Si el archivo curb no existe, cae a labels originales.
    """
    if use_curb:
        curb_file = CURB_LABELS_ROOT / seq / "labels" / f"{scan_id:06d}.label"
        if curb_file.exists():
            return curb_file
    return LABELS_ROOT / seq / "labels" / f"{scan_id:06d}.label"


def get_sequence_info(seq: str, use_curb: bool = True) -> dict:
    """
    Devuelve dict compatible con el formato SEQUENCES usado en run_pipeline_viz, tests, etc.

    Returns:
        dict con claves 'data_dir', 'label_dir', 'poses_file'
    """
    return {
        'data_dir': VELODYNE_ROOT / seq,
        'label_dir': get_labels_dir(seq, use_curb=use_curb),
        'poses_file': str(get_poses_file(seq)),
    }
