#!/usr/bin/env python3
"""
Parámetros del pipeline de detección de obstáculos LiDAR.

Perfiles predefinidos para distintos sensores y datasets.
Cada perfil devuelve un PipelineConfig con valores optimizados.

Uso:
    from pipeline_params import get_kitti_config, get_goose_config

    cfg = get_kitti_config()                  # KITTI defaults
    cfg = get_kitti_config(curb=True)         # KITTI con detección de bordillos
    cfg = get_goose_config(curb=True)         # GOOSE con detección de bordillos
"""

from lidar_pipeline_suite import PipelineConfig


# ================================================================================
# PERFILES DE SENSOR
# ================================================================================

# Velodyne HDL-64E (KITTI)
SENSOR_HDL64E = dict(
    sensor_height=1.73,
    min_range=2.7,
    max_range=80.0,
    n_rings=64,
    fov_up_deg=2.0,
    fov_down_deg=-24.33,
)

# Velodyne Alpha Prime VLS-128 (GOOSE)
SENSOR_VLS128 = dict(
    sensor_height=2.089,
    min_range=2.7,
    max_range=80.0,
    n_rings=128,
    fov_up_deg=15.0,
    fov_down_deg=-15.0,
)


# ================================================================================
# PARÁMETROS DE PATCHWORK++ (defaults C++ oficiales)
# ================================================================================

PATCHWORK_DEFAULTS = dict(
    num_zones=4,
    num_rings_each_zone=[2, 4, 4, 4],
    num_sectors_each_zone=[16, 32, 54, 32],
    num_iter=3,
    num_lpr=20,
    num_min_pts=10,
    th_dist=0.125,
    uprightness_thr=0.707,
)


# ================================================================================
# PARÁMETROS OPTIMIZADOS POR STAGE
# ================================================================================

# Wall Rejection — optimizado grid search con labels 3D-Curb (SemanticKITTI val seq 08)
WALL_REJECTION_OPTIMIZED = dict(
    enable_hybrid_wall_rejection=True,
    wall_rejection_slope=1.0,
    wall_height_diff_threshold=0.15,
    wall_kdtree_radius=0.2,
    wall_min_neighbors=5,
)

# Curb Detection — optimizado grid search con labels 3D-Curb
CURB_DETECTION_KITTI = dict(
    enable_curb_detection=True,
    curb_height_min=0.05,
    curb_height_max=0.30,
    curb_min_consecutive=2,
    curb_ring_gap=3,
)

# DBSCAN — optimizado grid search (SemanticKITTI)
DBSCAN_OPTIMIZED = dict(
    enable_cluster_filtering=True,
    cluster_eps=1.2,
    cluster_min_samples=12,
    cluster_min_pts=10,
)


# ================================================================================
# FUNCIONES DE CONFIGURACIÓN
# ================================================================================

def get_kitti_config(curb=False, dbscan=True, verbose=False) -> PipelineConfig:
    """Config optimizada para Velodyne HDL-64E + SemanticKITTI."""
    params = {
        **SENSOR_HDL64E,
        **PATCHWORK_DEFAULTS,
        **WALL_REJECTION_OPTIMIZED,
        **DBSCAN_OPTIMIZED,
        'enable_curb_detection': curb,
        'enable_cluster_filtering': dbscan,
        'verbose': verbose,
    }
    if curb:
        params.update(CURB_DETECTION_KITTI)
    return PipelineConfig(**params)


def get_goose_config(curb=False, dbscan=True, verbose=False) -> PipelineConfig:
    """Config para Velodyne Alpha Prime VLS-128 + GOOSE dataset.
    Usa los mismos parámetros de algoritmo que KITTI (generalización)."""
    params = {
        **SENSOR_VLS128,
        **PATCHWORK_DEFAULTS,
        **WALL_REJECTION_OPTIMIZED,
        **DBSCAN_OPTIMIZED,
        'enable_curb_detection': curb,
        'enable_cluster_filtering': dbscan,
        'verbose': verbose,
    }
    if curb:
        params.update(CURB_DETECTION_KITTI)
    return PipelineConfig(**params)
