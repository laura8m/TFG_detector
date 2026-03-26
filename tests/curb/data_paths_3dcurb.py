#!/usr/bin/env python3
"""
data_paths_3dcurb.py
====================
Módulo centralizado para construir rutas al dataset 3D-Curb.

El dataset 3D-Curb sigue el formato SemanticKITTI pero con menos frames
por secuencia y añade la etiqueta 3 (bordillo / curb).

Estructura velodyne (misma que SemanticKITTI):
    {velodyne_root}/dataset/sequences/{seq}/velodyne/{scan_id:06d}.bin

Estructura labels 3D-Curb:
    {labels_root}/{seq}/labels/{scan_id:06d}.label
"""

import sys
import numpy as np
from pathlib import Path

# Permitir importar módulos del directorio padre (sota_idea/)
_SOTA_DIR = Path(__file__).resolve().parents[2]
if str(_SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(_SOTA_DIR))

# ========================================
# Constantes de secuencias
# ========================================
TRAIN_SEQS = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQS = ['08']

# ========================================
# Etiquetas semánticas (SemanticKITTI + 3D-Curb)
# ========================================
CURB_LABEL = 3

IGNORE_LABELS = np.array([0, 1, 52, 99], dtype=np.uint32)

OBSTACLE_LABELS = np.array([
    CURB_LABEL,  # 3 - curb / bordillo
    10, 11, 13, 15, 16, 18, 20,  # vehículos + persona
    30, 31, 32,  # moving
    50, 51,  # building, fence
    70, 71,  # vegetation, trunk
    80, 81,  # fence, pole
    252, 253, 254, 255, 256, 257, 258, 259  # moving-*
], dtype=np.uint32)

GROUND_LABELS = np.array([40, 44, 48, 49, 60, 72], dtype=np.uint32)

# ========================================
# Búsqueda de raíces de datos
# ========================================
_BASE_DIR = Path(__file__).resolve().parents[2]  # sota_idea/
_HOME = Path.home()

# Usar los .bin de 3D-Curb (tienen menos puntos, coinciden con los labels)
_VELODYNE_CANDIDATES = [
    _BASE_DIR / "3d_curb_labels",
    _HOME / "laura" / "sota_ws" / "TFG_detector" / "3d_curb_labels",
]

_LABELS_CANDIDATES = [
    _BASE_DIR / "3d_curb_labels",
    _HOME / "laura" / "sota_ws" / "TFG_detector" / "3d_curb_labels",
]


def _find_root(candidates, description):
    for p in candidates:
        if p.is_dir():
            return p
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No se encontró directorio de {description}. Rutas probadas:\n  {tried}"
    )


def _velodyne_root():
    return _find_root(_VELODYNE_CANDIDATES, "velodyne (3D-Curb / SemanticKITTI)")


def _labels_root():
    return _find_root(_LABELS_CANDIDATES, "labels 3D-Curb")


# ========================================
# Funciones públicas
# ========================================

def get_velodyne_dir(seq, velodyne_root=None):
    root = Path(velodyne_root) if velodyne_root else _velodyne_root()
    return root / seq / "velodyne"


def get_labels_dir(seq, labels_root=None):
    root = Path(labels_root) if labels_root else _labels_root()
    return root / seq / "labels"


def get_scan_file(seq, scan_id, velodyne_root=None):
    return get_velodyne_dir(seq, velodyne_root) / f"{scan_id:06d}.bin"


def get_label_file(seq, scan_id, labels_root=None):
    return get_labels_dir(seq, labels_root) / f"{scan_id:06d}.label"


def discover_scan_ids(seq, stride=1, velodyne_root=None, labels_root=None):
    """
    Descubre scan IDs que existen tanto en velodyne como en labels.
    """
    vel_dir = get_velodyne_dir(seq, velodyne_root)
    lab_dir = get_labels_dir(seq, labels_root)

    if not vel_dir.is_dir():
        raise FileNotFoundError(f"Directorio velodyne no encontrado: {vel_dir}")
    if not lab_dir.is_dir():
        raise FileNotFoundError(f"Directorio labels no encontrado: {lab_dir}")

    vel_ids = {int(p.stem) for p in vel_dir.glob("*.bin")}
    lab_ids = {int(p.stem) for p in lab_dir.glob("*.label")}

    common = sorted(vel_ids & lab_ids)
    return common[::stride]


def load_scan(seq, scan_id, velodyne_root=None, labels_root=None):
    """
    Carga un scan completo: puntos + labels + máscaras.

    Returns:
        points: (N, 3) coordenadas XYZ
        gt_mask: (N,) máscara de obstáculos GT
        valid_mask: (N,) máscara de puntos válidos (no ignore)
        curb_mask_gt: (N,) máscara de puntos bordillo GT
        labels: (N,) labels raw
    """
    scan_file = get_scan_file(seq, scan_id, velodyne_root)
    label_file = get_label_file(seq, scan_id, labels_root)

    points = np.fromfile(str(scan_file), dtype=np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(str(label_file), dtype=np.uint32) & 0xFFFF

    gt_mask = np.isin(labels, OBSTACLE_LABELS)
    valid_mask = ~np.isin(labels, IGNORE_LABELS)
    curb_mask_gt = labels == CURB_LABEL

    return points, gt_mask, valid_mask, curb_mask_gt, labels
