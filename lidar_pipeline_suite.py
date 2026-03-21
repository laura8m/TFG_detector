#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import time

# ROS 2 (opcional, solo para publicación en RViz)
try:
    import rclpy
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ================================================================================
# CONFIGURACIÓN DEL PIPELINE
# ================================================================================

@dataclass
class PipelineConfig:
    """
    Configuración unificada del pipeline de detección de obstáculos LiDAR.

    Contiene todos los parámetros organizados por etapa del pipeline,
    con valores por defecto optimizados para Velodyne HDL-64E y dataset KITTI.
    Diseñada para facilitar ablation studies mediante banderas booleanas por etapa.

    Etapas:
    - Stage 1: Segmentación de suelo (Patchwork++) + rechazo de paredes
    - Stage 2: Detección de anomalías delta-r
    - Stage 3: Filtrado por clustering DBSCAN
    """

    # ========================================
    # STAGE 1: Segmentación de suelo + rechazo de paredes
    # ========================================

    # === Patchwork++ Base ===
    sensor_height: float = 1.73  # Altura del sensor sobre el suelo (m)
    min_range: float = 2.7  # Distancia mínima para filtrar ego-vehículo (m)
    max_range: float = 80.0  # Distancia máxima efectiva del sensor (m)
    num_zones: int = 4  # Número de zonas concéntricas (CZM)
    num_rings_each_zone: List[int] = field(default_factory=lambda: [2, 4, 4, 4])
    num_sectors_each_zone: List[int] = field(default_factory=lambda: [16, 32, 54, 32])
    num_iter: int = 3  # Iteraciones de refinamiento de plano
    num_lpr: int = 20  # Lowest Point Representative
    num_min_pts: int = 10  # Puntos mínimos por bin
    th_dist: float = 0.2  # Umbral de distancia para inliers (m)
    uprightness_thr: float = 0.707  # Umbral de verticalidad (cos(45°))

    # === Rechazo híbrido de paredes ===
    # Bandera de ablation: ¿Activar rechazo de paredes?
    enable_hybrid_wall_rejection: bool = True

    wall_rejection_slope: float = 0.9  # Umbral nz (normal vertical) — optimizado grid search
    # Si abs(n[2]) < 0.9 → plano inclinado → sospecha de pared

    wall_height_diff_threshold: float = 0.2  # Delta-Z local (20cm) — optimizado grid search
    # Si variación de altura en vecindad > 20cm → confirmada como pared

    wall_kdtree_radius: float = 0.3  # Radio de vecindad local (m) — optimizado grid search

    # ========================================
    # STAGE 2: Detección de anomalías delta-r
    # ========================================

    threshold_obs: float = -0.8  # Obstáculo positivo (m) — optimizado grid search conservador
    threshold_void: float = 1.5  # Void/depresión (m) — optimizado grid search conservador
    delta_r_conservative: bool = True  # Modo conservador: solo rescate, nunca degradar Stage 1
    delta_r_min_nz: float = 0.95  # nz mínimo para considerar bin fiable — optimizado grid search

    # ========================================
    # STAGE 3: Filtrado por clustering DBSCAN
    # ========================================

    enable_cluster_filtering: bool = True  # Activar/desactivar Stage 3
    cluster_eps: float = 1.2  # DBSCAN epsilon (m) — optimizado grid search (labels corregidas)
    cluster_min_samples: int = 12  # DBSCAN min_samples — optimizado grid search (labels corregidas)
    cluster_min_pts: int = 10  # Puntos mínimos por cluster — optimizado grid search (labels corregidas)

    # ========================================
    # GENERAL
    # ========================================

    verbose: bool = True  # Imprimir logs


# ================================================================================
# CLASE PRINCIPAL DEL PIPELINE
# ================================================================================

class LidarPipelineSuite:
    """
    Suite modular de procesamiento LiDAR 3D para detección de obstáculos.

    Implementa un pipeline secuencial de 3 etapas:
      1. Segmentación de suelo (Patchwork++ + rechazo de paredes)
      2. Detección de anomalías delta-r
      3. Filtrado por clustering DBSCAN (elimina ruido disperso)

    Cada etapa puede activarse/desactivarse mediante PipelineConfig para
    facilitar ablation studies.

    Estado compartido entre etapas:
    - self.local_planes: Planos locales por bin CZM para cálculo de delta-r
    Nota: El filtro temporal Bayesiano (Dewan et al.) se eliminó tras ablation
    study que demostró que no mejora resultados en KITTI (buen tiempo).
    Ver lidar_pipeline_suite_with_bayes.py para la versión con Bayes.
    """

    def __init__(
        self,
        config: PipelineConfig,
        data_path: str = None,
        ros_node=None
    ):
        """
        Inicializa el pipeline con la configuración dada.

        Args:
            config: Configuración del pipeline (parámetros por etapa)
            data_path: Ruta a datos KITTI (opcional)
            ros_node: Nodo ROS 2 para publicación (opcional)
        """
        # === CONFIGURACIÓN ===
        self.config = config
        self.data_path = Path(data_path) if data_path else None
        self.ros_node = ros_node

        # === ESTADO COMPARTIDO ===
        self.local_planes = {}  # Dict[(z,r,s)] -> (normal, d)
        # === INICIALIZAR SUBMÓDULOS ===
        self._init_patchwork()
        self.initialize_czm_params()

        if config.verbose:
            print("[INFO] LidarPipelineSuite inicializado")
            print(f"  - Rechazo de paredes: {'SI' if config.enable_hybrid_wall_rejection else 'NO'}")
            print(f"  - Filtrado por clustering: {'SI' if config.enable_cluster_filtering else 'NO'}")

    # ========================================
    # INICIALIZACIÓN DE SUBMÓDULOS
    # ========================================

    def _init_patchwork(self):
        """
        Inicializa Patchwork++ con los parámetros del config.

        Configura el segmentador de suelo con los umbrales CZM, distancia
        de inliers, verticalidad y demás parámetros de Patchwork++.
        """
        try:
            import pypatchworkpp
            self.params = pypatchworkpp.Parameters()
            self.params.verbose = False
            self.params.sensor_height = self.config.sensor_height
            self.params.min_range = self.config.min_range
            self.params.max_range = self.config.max_range
            self.params.num_iter = self.config.num_iter
            self.params.num_lpr = self.config.num_lpr
            self.params.num_min_pts = self.config.num_min_pts
            self.params.th_dist = self.config.th_dist
            self.params.uprightness_thr = self.config.uprightness_thr
            self.params.adaptive_seed_selection_margin = -1.1
            self.params.enable_RNR = False

            # Parámetros CZM
            self.params.num_zones = self.config.num_zones
            self.params.num_rings_each_zone = self.config.num_rings_each_zone
            self.params.num_sectors_each_zone = self.config.num_sectors_each_zone

            self.patchwork = pypatchworkpp.patchworkpp(self.params)

            if self.config.verbose:
                print("[INFO] Patchwork++ inicializado")
        except ImportError:
            print("[ERROR] pypatchworkpp no encontrado")
            print("  Instalar desde: /home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python")
            sys.exit(1)

    def initialize_czm_params(self):
        """
        Inicializa parámetros del modelo de zonas concéntricas (CZM).

        CZM divide el espacio polar en 4 zonas radiales con diferente
        resolución (anillas x sectores). Calcula los límites de cada zona,
        el tamaño de anilla (m/ring) y el tamaño de sector (rad/sector)
        necesarios para asignar cada punto a su bin correspondiente.
        """
        min_r = self.config.min_range
        max_r = self.config.max_range

        # Límites de zonas (fórmula de Patchwork++)
        self.min_ranges = [
            min_r,
            (7 * min_r + max_r) / 8.0,
            (3 * min_r + max_r) / 4.0,
            (min_r + max_r) / 2.0
        ]

        # Tamaños de anillos (m/ring)
        self.ring_sizes = []
        for i in range(4):
            end_r = self.min_ranges[i+1] if i < 3 else max_r
            self.ring_sizes.append(
                (end_r - self.min_ranges[i]) / self.config.num_rings_each_zone[i]
            )

        # Tamaños de sectores (radianes/sector)
        self.sector_sizes = [
            2 * np.pi / n for n in self.config.num_sectors_each_zone
        ]

        if self.config.verbose:
            print(f"[INFO] CZM inicializado: {sum(self.config.num_rings_each_zone)} rings, "
                  f"{sum([r*s for r, s in zip(self.config.num_rings_each_zone, self.config.num_sectors_each_zone)])} bins totales")

    # ========================================
    # CZM: CÁLCULO DE BIN (zona, anilla, sector)
    # ========================================

    def get_czm_bin(self, x, y):
        """
        Calcula bin CZM (zona, anilla, sector) para cada punto (vectorizado).

        Asigna cada punto a su bin correspondiente en el modelo de zonas
        concéntricas usando coordenadas polares. Puntos fuera del rango
        válido reciben índice -1.

        Args:
            x: (N,) coordenadas X
            y: (N,) coordenadas Y

        Returns:
            zone_idx: (N,) índices de zona [0-3] (-1 si inválido)
            ring_idx: (N,) índices de anilla [0-3] (-1 si inválido)
            sector_idx: (N,) índices de sector [0-53] (-1 si inválido)
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)

        N = len(r)
        zone_idx = np.full(N, -1, dtype=np.int32)
        ring_idx = np.full(N, -1, dtype=np.int32)
        sector_idx = np.full(N, -1, dtype=np.int32)

        # Usar digitize para asignar zonas de golpe
        zone_bounds = [self.min_ranges[0], self.min_ranges[1], self.min_ranges[2], self.min_ranges[3], self.config.max_range + 0.01]
        zone_raw = np.digitize(r, zone_bounds) - 1  # 0-3 para válidos
        valid = (zone_raw >= 0) & (zone_raw <= 3)

        if np.any(valid):
            vi = np.where(valid)[0]
            z_v = zone_raw[vi]
            zone_idx[vi] = z_v

            # Arrays de parámetros por zona para lookup vectorizado
            min_ranges_arr = np.array(self.min_ranges, dtype=np.float64)
            ring_sizes_arr = np.array(self.ring_sizes, dtype=np.float64)
            sector_sizes_arr = np.array(self.sector_sizes, dtype=np.float64)
            max_rings_arr = np.array(self.config.num_rings_each_zone, dtype=np.int32)
            max_sectors_arr = np.array(self.config.num_sectors_each_zone, dtype=np.int32)

            # Vectorizado: ring y sector para todos los puntos válidos a la vez
            r_base = min_ranges_arr[z_v]
            r_size = ring_sizes_arr[z_v]
            s_size = sector_sizes_arr[z_v]

            ring_idx[vi] = np.clip(((r[vi] - r_base) / r_size).astype(np.int32), 0, max_rings_arr[z_v] - 1)
            sector_idx[vi] = np.clip((theta[vi] / s_size).astype(np.int32), 0, max_sectors_arr[z_v] - 1)

        return zone_idx, ring_idx, sector_idx

    def get_czm_bin_scalar(self, x, y):
        """
        Versión escalar de get_czm_bin para un solo punto.

        Útil para construcción de tablas de lookup donde se procesa
        un centro de bin a la vez.

        Args:
            x: coordenada X (escalar)
            y: coordenada Y (escalar)

        Returns:
            (zone, ring, sector) tupla o None si fuera de rango
        """
        r = np.sqrt(x**2 + y**2)
        if r <= self.config.min_range or r > self.config.max_range:
            return None

        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi

        # Determinar zona
        if r < self.min_ranges[1]:
            z = 0
            r_base = self.min_ranges[0]
        elif r < self.min_ranges[2]:
            z = 1
            r_base = self.min_ranges[1]
        elif r < self.min_ranges[3]:
            z = 2
            r_base = self.min_ranges[2]
        else:
            z = 3
            r_base = self.min_ranges[3]

        r_idx = int((r - r_base) / self.ring_sizes[z])
        r_idx = min(r_idx, self.config.num_rings_each_zone[z] - 1)
        r_idx = max(r_idx, 0)

        s_idx = int(theta / self.sector_sizes[z])
        s_idx = min(s_idx, self.config.num_sectors_each_zone[z] - 1)
        s_idx = max(s_idx, 0)

        return (z, r_idx, s_idx)

    # ========================================
    # STAGE 1: SEGMENTACIÓN DE SUELO + RECHAZO DE PAREDES
    # ========================================

    @staticmethod
    def _validate_and_reject_walls_pointwise(points, ground_indices,
                                              delta_z_threshold=0.3,
                                              use_percentiles=True,
                                              kdtree_radius=0.5,
                                              min_neighbors=5):
        """
        Rechazo híbrido de paredes: filtro rápido por bins + refinamiento punto a punto.

        Patchwork++ clasifica como suelo la base de paredes y objetos verticales.
        Este método detecta y rechaza esos puntos mal clasificados usando dos fases:

        Fase 1 - Filtro por bins (rápido):
            Agrupa puntos ground en bins CZM (zona/anilla/sector) y marca como
            sospechosos aquellos bins con variación vertical delta-Z > umbral.

        Fase 2 - Refinamiento punto a punto (solo bins sospechosos):
            Para cada punto sospechoso, analiza la vecindad local con KDTree.
            Solo rechaza puntos individuales con delta-Z local > umbral, evitando
            rechazar bins completos (reduce falsos positivos).

        Args:
            points: Nube completa (N, 3)
            ground_indices: Índices de puntos clasificados como suelo
            delta_z_threshold: Umbral de variación vertical (m)
            use_percentiles: Usar percentiles 95/5 en vez de rango completo
            kdtree_radius: Radio de búsqueda local (m)
            min_neighbors: Mínimo de vecinos para validar estadística

        Returns:
            Índices de puntos rechazados (paredes detectadas)
        """
        if len(ground_indices) == 0:
            return np.array([], dtype=np.int32)

        ground_pts = points[ground_indices]

        # --- Fase 1: Agrupar puntos ground por bins CZM ---
        x = ground_pts[:, 0]
        y = ground_pts[:, 1]
        z = ground_pts[:, 2]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)

        # Bins CZM: 4 zonas concéntricas, cada una dividida en anillas y sectores
        zone_boundaries = [2.7, 30.0, 50.0, 80.0]
        zone_idx = np.digitize(r, zone_boundaries) - 1
        zone_idx = np.clip(zone_idx, 0, 3)

        # Anilla y sector por zona (simplificado: 32 sectores uniformes)
        n_sectors = 32
        sector_idx = (theta / (2 * np.pi) * n_sectors).astype(np.int32)
        sector_idx = np.clip(sector_idx, 0, n_sectors - 1)

        # Anilla basada en distancia radial dentro de cada zona
        ring_idx = np.zeros(len(ground_pts), dtype=np.int32)
        for zone_id in range(4):
            mask = zone_idx == zone_id
            if np.any(mask):
                r_zone = r[mask]
                r_min = zone_boundaries[zone_id]
                r_max = zone_boundaries[zone_id+1] if zone_id < 3 else 80.0
                # 4 anillas por zona
                ring_idx[mask] = ((r_zone - r_min) / (r_max - r_min) * 4).astype(np.int32)
                ring_idx[mask] = np.clip(ring_idx[mask], 0, 3)

        # Crear bin_id único: (zone * 128) + (ring * 32) + sector
        bin_id = zone_idx * 128 + ring_idx * 32 + sector_idx

        # --- Fase 2: Identificar bins con variación vertical sospechosa (vectorizado) ---

        # Agrupar por bin_id usando sort + reduceat
        sort_bin = np.argsort(bin_id)
        sorted_bins = bin_id[sort_bin]
        sorted_z_bins = z[sort_bin]

        unique_bins, bin_starts = np.unique(sorted_bins, return_index=True)
        bin_ends = np.append(bin_starts[1:], len(sorted_bins))
        bin_counts = bin_ends - bin_starts

        # Ordenar z dentro de cada grupo para percentiles rápidos
        sorted_z_for_pct = sorted_z_bins.copy()
        for i in range(len(bin_starts)):
            s, e = bin_starts[i], bin_ends[i]
            sorted_z_for_pct[s:e] = np.sort(sorted_z_for_pct[s:e])

        # P5/P95 por bin (vectorizado)
        enough_mask = bin_counts >= min_neighbors
        if use_percentiles:
            p5_idx = np.clip((0.05 * (bin_counts - 1)).astype(np.int32), 0, bin_counts - 1)
            p95_idx = np.clip((0.95 * (bin_counts - 1)).astype(np.int32), 0, bin_counts - 1)
            bin_p5 = sorted_z_for_pct[bin_starts + p5_idx]
            bin_p95 = sorted_z_for_pct[bin_starts + p95_idx]
            bin_delta_z = bin_p95 - bin_p5
        else:
            # np.minimum.reduceat/maximum.reduceat
            bin_z_max = np.maximum.reduceat(sorted_z_bins, bin_starts)
            bin_z_min = np.minimum.reduceat(sorted_z_bins, bin_starts)
            bin_delta_z = bin_z_max - bin_z_min

        suspect_mask_bins = enough_mask & (bin_delta_z > delta_z_threshold)

        # Recoger puntos sospechosos (vectorizado)
        suspect_indices_list = []
        suspect_bin_indices = np.where(suspect_mask_bins)[0]
        for bi in suspect_bin_indices:
            s, e = bin_starts[bi], bin_ends[bi]
            suspect_indices_list.append(sort_bin[s:e])

        # --- Fase 3: Refinamiento punto a punto con voxel grid (O(N)) ---

        rejected_mask = np.zeros(len(ground_pts), dtype=bool)

        if len(suspect_indices_list) > 0:
            suspect_indices = np.concatenate(suspect_indices_list)

            if len(suspect_indices) > 0:
                # Voxel grid 2D (XY): celdas de 1.0m (= diámetro del radio KDTree)
                # Cada celda tiene ~20-40 puntos → percentiles robustos sin vecinos
                cell_size = kdtree_radius * 2.0
                vox_x = np.floor(ground_pts[:, 0] / cell_size).astype(np.int64)
                vox_y = np.floor(ground_pts[:, 1] / cell_size).astype(np.int64)
                gz = ground_pts[:, 2]

                # Hash espacial → clave única por celda
                voxel_key = vox_x * 100003 + vox_y

                # Ordenar por (voxel_key, z) para obtener z ordenado dentro de grupos
                # Doble sort: primero por key, luego estable por z dentro de cada key
                sort_idx = np.lexsort((gz, voxel_key))
                sorted_keys = voxel_key[sort_idx]
                sorted_z = gz[sort_idx]

                # Límites de cada grupo (voxel)
                change_idx = np.where(np.diff(sorted_keys) != 0)[0] + 1
                group_starts = np.concatenate([[0], change_idx])
                group_ends = np.concatenate([change_idx, [len(sorted_keys)]])
                unique_keys = sorted_keys[group_starts]
                counts = group_ends - group_starts

                # P5 y P95 por voxel (z ya ordenado dentro de cada grupo por lexsort)
                p5_idx = np.clip((0.05 * (counts - 1)).astype(np.int32), 0, counts - 1)
                p95_idx = np.clip((0.95 * (counts - 1)).astype(np.int32), 0, counts - 1)
                voxel_p5 = sorted_z[group_starts + p5_idx]
                voxel_p95 = sorted_z[group_starts + p95_idx]

                # Lookup vectorizado: cada sospechoso → su celda (sin vecinos)
                suspect_keys = vox_x[suspect_indices] * 100003 + vox_y[suspect_indices]

                positions = np.searchsorted(unique_keys, suspect_keys)
                positions = np.clip(positions, 0, len(unique_keys) - 1)
                valid = unique_keys[positions] == suspect_keys

                # Decisión vectorizada: P95-P5 > umbral y suficientes puntos
                delta_z_local = np.where(valid, voxel_p95[positions] - voxel_p5[positions], 0.0)
                enough_pts = np.where(valid, counts[positions] >= min_neighbors, False)

                rejected_suspects = enough_pts & (delta_z_local > delta_z_threshold)
                rejected_mask[suspect_indices[rejected_suspects]] = True

        rejected_indices = ground_indices[rejected_mask]
        return rejected_indices if len(rejected_indices) > 0 else np.array([], dtype=np.int32)

    def segment_ground(self, points: np.ndarray):
        """
        Segmentación ground/nonground con Patchwork++ y rechazo de paredes punto a punto.

        Pipeline interno:
        1. Ejecutar Patchwork++ para obtener ground/nonground base
        2. Extraer planos locales (centros, normales) de bins CZM y filtrar
           planos con normal vertical insuficiente (nz < umbral → pared)
        3. Rechazo de paredes punto a punto (si habilitado):
           Analiza cada punto ground individualmente con KDTree,
           rechazando puntos con escalón vertical local > umbral
        4. Construir normales y distancia de plano per-point (vectorizado)

        Args:
            points: (N, 3) array de puntos XYZ

        Returns:
            ground_points: (M, 3) puntos ground sin paredes
            n_per_point: (N, 3) normales por punto
            d_per_point: (N,) distancias plano por punto
            rejected_indices: (W,) índices de wall points rechazados (ground-space)
        """
        # 1. Ejecutar Patchwork++
        self.patchwork.estimateGround(points)

        ground_points = self.patchwork.getGround()
        ground_indices = self.patchwork.getGroundIndices()
        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()

        # 2. Construir lookup de planos locales (vectorizado)
        local_planes = {}

        if len(centers) > 0:
            with np.errstate(invalid='ignore'):
                centers_arr = np.asarray(centers, dtype=np.float64)
                normals_arr = np.asarray(normals, dtype=np.float64)

            # Asegurar normales apunten hacia arriba
            flip = normals_arr[:, 2] < 0
            normals_arr[flip] *= -1

            # Filtrar planos con nz suficiente (no paredes)
            valid_nz = normals_arr[:, 2] >= self.config.wall_rejection_slope

            # Obtener bins vectorizado
            z_idx, r_idx, s_idx = self.get_czm_bin(centers_arr[:, 0], centers_arr[:, 1])
            valid_bin = (z_idx >= 0) & valid_nz

            # Calcular d vectorizado: d = -(n . c)
            d_arr = -np.sum(normals_arr * centers_arr, axis=1)

            # Construir dict (solo ~400 bins, loop rápido sobre resultados filtrados)
            valid_indices = np.where(valid_bin)[0]
            for i in valid_indices:
                bin_id = (int(z_idx[i]), int(r_idx[i]), int(s_idx[i]))
                local_planes[bin_id] = (normals_arr[i].astype(np.float32), float(d_arr[i]))

        # Almacenar para otras etapas
        self.local_planes = local_planes

        # 3. Rechazo de paredes punto a punto
        rejected_wall_indices = np.array([], dtype=np.int32)

        if self.config.enable_hybrid_wall_rejection and len(ground_indices) > 0:
            rejected_wall_indices = self._validate_and_reject_walls_pointwise(
                points,
                ground_indices,
                delta_z_threshold=self.config.wall_height_diff_threshold,
                use_percentiles=True,
                kdtree_radius=self.config.wall_kdtree_radius,
                min_neighbors=5
            )

        if self.config.verbose:
            print(f"[Stage 1] Planos locales: {len(local_planes)}")
            print(f"[Stage 1] Rechazo de paredes: {len(rejected_wall_indices)} puntos rechazados")

        # 4. Construir normales y d per-point (lookup vectorizado)
        n_per_point, d_per_point = self._build_per_point_plane_params(
            points, local_planes
        )

        return ground_points, n_per_point, d_per_point, rejected_wall_indices

    def _build_per_point_plane_params(
        self,
        points: np.ndarray,
        local_planes: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye normales y distancia de plano por punto usando lookup de planos locales.

        Para cada punto, determina su bin CZM y asigna la normal y distancia del
        plano local correspondiente. Si no existe plano local para ese bin, usa
        el plano global por defecto (horizontal a altura del sensor).

        Args:
            points: (N, 3) puntos XYZ
            local_planes: Dict[(z,r,s)] -> (normal, d)

        Returns:
            n_per_point: (N, 3) normales por punto
            d_per_point: (N,) distancias plano por punto
        """
        # Plano global por defecto (alternativa)
        global_normal = np.array([0.0, 0.0, 1.0])
        global_d = self.config.sensor_height

        # Crear tabla de lookup (4 zonas, 4 anillas, 54 sectores máx)
        planes_table = np.zeros((4, 4, 54, 4), dtype=np.float32)
        planes_table[..., :3] = global_normal
        planes_table[..., 3] = global_d

        # Rellenar con planos locales
        for bin_id, (n_loc, d_loc) in local_planes.items():
            z_b, r_b, s_b = bin_id
            if 0 <= z_b < 4 and 0 <= r_b < 4 and 0 <= s_b < 54:
                planes_table[z_b, r_b, s_b, :3] = n_loc
                planes_table[z_b, r_b, s_b, 3] = d_loc

        # Obtener bins para todos los puntos (vectorizado)
        z_idx, r_idx, s_idx = self.get_czm_bin(points[:, 0], points[:, 1])

        # Inicializar con plano global
        n_per_point = np.full((len(points), 3), global_normal, dtype=np.float32)
        d_per_point = np.full(len(points), global_d, dtype=np.float32)

        # Buscar bins válidos
        valid_bins = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)

        if np.any(valid_bins):
            z_v = z_idx[valid_bins]
            r_v = r_idx[valid_bins]
            s_v = s_idx[valid_bins]

            plane_params = planes_table[z_v, r_v, s_v]
            n_per_point[valid_bins] = plane_params[:, :3]
            d_per_point[valid_bins] = plane_params[:, 3]

        return n_per_point, d_per_point

    def stage1_complete(self, points: np.ndarray) -> Dict:
        """
        Ejecuta Stage 1 completo: segmentación de suelo + rechazo de paredes.

        Orquesta las dos sub-etapas de Stage 1:
        1. Patchwork++ para segmentación base ground/nonground
        2. Rechazo híbrido de paredes punto a punto (si habilitado)

        Las paredes rechazadas se reclasifican como nonground (son obstáculos).

        Args:
            points: (N, 3) array de puntos XYZ

        Returns:
            dict con:
            - ground_indices: (M,) índices ground limpios (sin paredes)
            - nonground_indices: (K,) índices nonground (incluyendo paredes)
            - local_planes: Dict de planos locales por bin CZM
            - rejected_walls: (W,) índices de paredes rechazadas
            - n_per_point, d_per_point: normales y distancias per-point
            - timing_ms: tiempo de ejecución (ms)
        """
        t0 = time.time()

        # 1. Patchwork++ + rechazo de paredes punto a punto
        ground_pts, n_per_point, d_per_point, rejected_wall_indices = \
            self.segment_ground(points)

        # 2. Obtener índices
        ground_indices = self.patchwork.getGroundIndices()
        nonground_indices = self.patchwork.getNongroundIndices()

        # Filtrar ground removiendo wall points
        # Las paredes pasan a nonground (son obstáculos, no desaparecen)
        if len(rejected_wall_indices) > 0:
            # Máscara booleana sobre punto global O(N) — más rápido que np.isin para N grande
            rejected_mask_full = np.zeros(len(points), dtype=bool)
            rejected_mask_full[rejected_wall_indices] = True
            clean_ground = ground_indices[~rejected_mask_full[ground_indices]]
            nonground_indices = np.concatenate([nonground_indices, rejected_wall_indices])
        else:
            clean_ground = ground_indices

        # Almacenar estado
        self.rejected_wall_indices = rejected_wall_indices

        timing = (time.time() - t0) * 1000  # ms

        if self.config.verbose:
            print(f"[Stage 1 Completo] {timing:.1f} ms")
            print(f"  Ground: {len(clean_ground)} | Paredes: {len(rejected_wall_indices)}")

        return {
            'ground_indices': clean_ground,
            'nonground_indices': nonground_indices,
            'local_planes': self.local_planes,
            'rejected_walls': rejected_wall_indices,
            'n_per_point': n_per_point,
            'd_per_point': d_per_point,
            'timing_ms': timing
        }

    # ========================================
    # STAGE 2: DETECCIÓN DE ANOMALÍAS DELTA-R
    # ========================================

    def compute_delta_r(
        self,
        points: np.ndarray,
        ground_indices: np.ndarray,
        n_per_point: np.ndarray,
        d_per_point: np.ndarray,
        nonground_indices: np.ndarray = None,
    ) -> Dict:
        """
        Stage 2: Detección de anomalías delta-r.

        Mide la desviación entre el rango medido y el rango esperado por el
        plano local:
            delta_r = r_medido - r_esperado

        Donde r_esperado se calcula proyectando el rayo sobre el plano local:
            r_esperado = -d / (n . dirección_rayo)

        Clasificación:
            delta_r < umbral_obs → Obstáculo positivo (más cerca que el plano)
            delta_r > umbral_void → Void/depresión (más lejos que el plano)
            Intermedio → Ground normal

        Modo conservador (delta_r_conservative=True):
            - Solo aplica delta-r en bins con nz >= delta_r_min_nz (plano fiable)
            - Nunca reclasifica non-ground de Stage 1 como ground
            - Solo permite ground→obstáculo o ground→void (rescate)

        Args:
            points: (N, 3) todos los puntos
            ground_indices: (M,) índices de puntos ground
            n_per_point: (N, 3) normales por punto
            d_per_point: (N,) offsets del plano por punto
            nonground_indices: (K,) índices nonground de Stage 1 (para modo conservador)

        Returns:
            Dict con:
                - delta_r: (N,) desviación de rango
                - likelihood: (N,) log-likelihood por punto
                - obs_mask: (N,) máscara booleana de obstáculos
                - void_mask: (N,) máscara booleana de voids
                - ground_mask: (N,) máscara booleana de ground
                - uncertain_mask: (N,) máscara de puntos inciertos
                - timing_ms: tiempo de ejecución
        """
        t_start = time.time()

        N = len(points)

        # ========================================
        # 1. CALCULAR r_medido y dirección del rayo
        # ========================================
        r_measured = np.linalg.norm(points, axis=1)

        # ========================================
        # 2. CALCULAR r_esperado y delta_r (fusionado, sin arrays intermedios)
        # ========================================
        # dot_prod = (point / |point|) . normal = (point . normal) / |point|
        dot_prod = np.einsum('ij,ij->i', points, n_per_point) / np.maximum(r_measured, 1e-6)

        # r_expected = -d / dot_prod (solo donde dot_prod < 0)
        # delta_r = r_measured - r_expected = r_measured + d / dot_prod
        safe_dot = np.where(dot_prod < -1e-3, dot_prod, -1e-3)
        delta_r = np.clip(r_measured + d_per_point / safe_dot, -20.0, 10.0)

        # ========================================
        # 3. CLASIFICACIÓN
        # ========================================
        threshold_obs = self.config.threshold_obs
        threshold_void = self.config.threshold_void

        if self.config.delta_r_conservative and nonground_indices is not None:
            # --- MODO CONSERVADOR ---
            # Empezar con la clasificación de Stage 1 (non-ground = obstáculo)
            obs_mask_final = np.zeros(N, dtype=bool)
            obs_mask_final[nonground_indices] = True  # Preservar Stage 1

            # Máscara de bins fiables: nz >= delta_r_min_nz
            nz_per_point = np.abs(n_per_point[:, 2])
            reliable_bin = nz_per_point >= self.config.delta_r_min_nz

            # Solo en puntos ground + bin fiable: rescatar como obstáculo o void
            ground_mask_s1 = np.ones(N, dtype=bool)
            ground_mask_s1[nonground_indices] = False
            rescatable = ground_mask_s1 & reliable_bin

            # Rescate: ground→obstáculo si delta-r indica anomalía
            rescued_obs = rescatable & (delta_r < threshold_obs)
            rescued_void = rescatable & (delta_r > threshold_void)
            obs_mask_final |= rescued_obs | rescued_void

            void_mask_final = np.zeros(N, dtype=bool)
            void_mask_final[nonground_indices[delta_r[nonground_indices] > threshold_void]] = True
            void_mask_final |= rescued_void

            n_rescued = int(rescued_obs.sum() + rescued_void.sum())
            if self.config.verbose:
                n_reliable = int(reliable_bin.sum())
                print(f"  [Delta-r conservador] Bins fiables: {n_reliable}/{N} pts | Rescatados: {n_rescued}")
        else:
            # --- MODO ORIGINAL ---
            obs_mask_final = (delta_r < threshold_obs) | (delta_r > threshold_void)
            void_mask_final = delta_r > threshold_void

        # Forzar paredes rechazadas como obstáculos
        if hasattr(self, 'rejected_wall_indices') and len(self.rejected_wall_indices) > 0:
            valid_idx = self.rejected_wall_indices[self.rejected_wall_indices < N]
            obs_mask_final[valid_idx] = True

        ground_mask_final = ~obs_mask_final

        # Likelihood (solo para compatibilidad, sin coste extra)
        likelihood_final = np.where(obs_mask_final, 2.0,
                           np.where(void_mask_final, 1.5, -2.0)).astype(np.float32)

        t_end = time.time()
        timing_ms = (t_end - t_start) * 1000.0

        if self.config.verbose:
            print(f"[Stage 2 Completo] {timing_ms:.1f} ms")
            print(f"  Obstáculos: {obs_mask_final.sum()} | Voids: {void_mask_final.sum()} | Ground: {ground_mask_final.sum()}")

        return {
            'delta_r': delta_r,
            'likelihood': likelihood_final,
            'obs_mask': obs_mask_final,
            'void_mask': void_mask_final,
            'ground_mask': ground_mask_final,
            'uncertain_mask': np.zeros(N, dtype=bool),
            'timing_ms': timing_ms
        }

    def stage2_complete(self, points: np.ndarray) -> Dict:
        """
        Stage 2 completo: ejecuta Stage 1 + delta-r.

        Orquesta la ejecución secuencial de Stage 1 (segmentación) y
        Stage 2 (detección de anomalías), reutilizando las normales y
        distancias de plano calculadas en Stage 1.

        Args:
            points: (N, 3) nube de puntos

        Returns:
            Dict con resultados combinados de Stage 1 + Stage 2
        """
        # Stage 1: Segmentación de suelo + rechazo de paredes
        stage1_result = self.stage1_complete(points)

        # Stage 2: delta-r
        stage2_result = self.compute_delta_r(
            points=points,
            ground_indices=stage1_result['ground_indices'],
            n_per_point=stage1_result['n_per_point'],
            d_per_point=stage1_result['d_per_point'],
            nonground_indices=stage1_result['nonground_indices'],
        )

        # Combinar resultados
        return {
            **stage1_result,  # ground_indices, rejected_walls, timing_ms (Stage 1)
            **stage2_result,  # delta_r, likelihood, obs_mask, void_mask, timing_ms (Stage 2)
            'timing_total_ms': stage1_result['timing_ms'] + stage2_result['timing_ms']
        }

    # ========================================
    # STAGE 3: FILTRADO POR CLUSTERING DBSCAN
    # ========================================

    def stage3_cluster_filtering(self, points: np.ndarray, stage2_result: Dict) -> Dict:
        """
        Stage 3: Filtrado de obstáculos por clustering DBSCAN.

        Los obstáculos reales (coches, edificios, personas) forman clusters densos
        en el espacio 3D. Los falsos positivos (ground ambiguo, ruido)
        son puntos dispersos sin estructura espacial coherente.

        DBSCAN agrupa puntos por densidad espacial:
        - Cluster grande (>= min_pts puntos) → obstáculo real → SE MANTIENE
        - Cluster pequeño (< min_pts) o ruido (sin cluster) → FP probable → SE ELIMINA

        Args:
            points: (N, 3) nube de puntos completa
            stage2_result: Dict de stage2 con obs_mask, likelihood, etc.

        Returns:
            Dict actualizado con obs_mask filtrado + info de clusters
        """
        t_start = time.time()

        cfg = self.config
        obs_mask = stage2_result['obs_mask'].copy()
        likelihood = stage2_result.get('likelihood', np.zeros(len(points))).copy()
        N = len(points)

        if not cfg.enable_cluster_filtering:
            return {
                **stage2_result,
                'cluster_labels': np.full(N, -1, dtype=np.int32),
                'n_clusters': 0,
                'n_noise_removed': 0,
                'timing_stage3_ms': 0.0,
            }

        # ========================================
        # 1. EXTRAER PUNTOS OBSTÁCULO
        # ========================================
        obs_indices = np.where(obs_mask)[0]
        n_obs = len(obs_indices)

        if n_obs == 0:
            if cfg.verbose:
                print(f"[Stage 3] Sin obstáculos. Saltar.")
            return {
                **stage2_result,
                'cluster_labels': np.full(N, -1, dtype=np.int32),
                'n_clusters': 0,
                'n_noise_removed': 0,
                'timing_stage3_ms': (time.time() - t_start) * 1000.0,
            }

        obs_pts = points[obs_indices]  # (M, 3)

        # ========================================
        # 2. VOXEL DOWNSAMPLING + DBSCAN
        # ========================================
        t_dbscan = time.time()

        # Voxelizar puntos obstáculo para reducir N antes de DBSCAN
        voxel_size = cfg.cluster_eps * 0.3  # Celdas más finas que eps para no perder resolución
        vox_coords = np.floor(obs_pts / voxel_size).astype(np.int32)
        # Hash único por voxel
        vox_keys = (vox_coords[:, 0].astype(np.int64) * 1000003 +
                    vox_coords[:, 1].astype(np.int64) * 1009 +
                    vox_coords[:, 2].astype(np.int64))

        # Agrupar puntos por voxel: calcular centroide de cada voxel
        unique_keys, inverse, counts = np.unique(vox_keys, return_inverse=True, return_counts=True)
        n_voxels = len(unique_keys)

        # Centroides con bincount (evita add.at, ~3x más rápido)
        voxel_centroids = np.column_stack([
            np.bincount(inverse, weights=obs_pts[:, d], minlength=n_voxels)
            for d in range(3)
        ]) / counts[:, np.newaxis]
        voxel_centroids = voxel_centroids.astype(np.float32)

        # DBSCAN sobre centroides (mucho menos puntos)
        db = DBSCAN(
            eps=cfg.cluster_eps,
            min_samples=max(2, cfg.cluster_min_samples // 2),
            n_jobs=-1
        )
        voxel_labels = db.fit_predict(voxel_centroids)

        # Propagar etiquetas de voxel a puntos originales
        cluster_labels_obs = voxel_labels[inverse]

        t_dbscan_end = time.time()

        # ========================================
        # 3. FILTRAR CLUSTERS PEQUEÑOS (vectorizado con bincount)
        # ========================================
        t_filter = time.time()

        # Contar puntos REALES por cluster usando bincount
        max_label = cluster_labels_obs.max()
        if max_label >= 0:
            # bincount solo funciona con >= 0, separar ruido (-1)
            cluster_sizes = np.bincount(cluster_labels_obs[cluster_labels_obs >= 0],
                                         minlength=max_label + 1)
            # Lookup vectorizado: cada punto → tamaño de su cluster
            point_cluster_size = np.where(
                cluster_labels_obs >= 0,
                cluster_sizes[cluster_labels_obs.clip(0)],
                0
            )
            valid_mask_obs = point_cluster_size >= cfg.cluster_min_pts
        else:
            valid_mask_obs = np.zeros(n_obs, dtype=bool)

        # Actualizar obs_mask
        obs_mask_new = np.zeros(N, dtype=bool)
        obs_mask_new[obs_indices[valid_mask_obs]] = True

        # Propagar al array completo
        cluster_labels_full = np.full(N, -1, dtype=np.int32)
        cluster_labels_full[obs_indices] = cluster_labels_obs

        t_filter_end = time.time()

        # ========================================
        # 4. MÉTRICAS
        # ========================================
        t_end = time.time()
        timing_ms = (t_end - t_start) * 1000.0

        n_noise = int((cluster_labels_obs == -1).sum())
        n_small_cluster = int((~valid_mask_obs & (cluster_labels_obs >= 0)).sum())
        n_removed = int(obs_mask.sum() - obs_mask_new.sum())

        if max_label >= 0:
            n_clusters_total = int((cluster_sizes > 0).sum())
            n_clusters_valid = int((cluster_sizes >= cfg.cluster_min_pts).sum())
        else:
            n_clusters_total = 0
            n_clusters_valid = 0

        if cfg.verbose:
            print(f"[Stage 3 DBSCAN] {timing_ms:.1f} ms")
            print(f"  Clusters: {n_clusters_total} total | {n_clusters_valid} válidos (>={cfg.cluster_min_pts} pts) | {n_clusters_total - n_clusters_valid} rechazados")
            print(f"  Puntos eliminados: {n_removed} ({100*n_removed/max(n_obs,1):.1f}%) — ruido: {n_noise}, clusters pequeños: {n_small_cluster}")
            print(f"  Tiempos: DBSCAN={1000*(t_dbscan_end-t_dbscan):.0f}ms | filtro={1000*(t_filter_end-t_filter):.0f}ms")

        return {
            **stage2_result,
            'obs_mask': obs_mask_new,
            'cluster_labels': cluster_labels_full,
            'n_clusters': n_clusters_valid,
            'n_clusters_rejected': n_clusters_total - n_clusters_valid,
            'n_noise_removed': n_noise,
            'n_small_cluster_removed': n_small_cluster,
            'n_cluster_total_removed': n_removed,
            'timing_stage3_ms': timing_ms,
            'timing_total_ms': stage2_result.get('timing_total_ms', 0) + timing_ms,
        }

    def stage3_complete(self, points: np.ndarray) -> Dict:
        """
        Pipeline completo: ejecuta Stages 1-2 + filtrado DBSCAN.

        Args:
            points: (N, 3) nube de puntos

        Returns:
            Dict con resultados de Stages 1-3
        """
        stage2_result = self.stage2_complete(points)
        self.last_stage2_result = stage2_result
        return self.stage3_cluster_filtering(points, stage2_result)

    # ========================================
    # UTILIDADES: EGOMOTION Y POSES
    # ========================================

    @staticmethod
    def load_kitti_poses(poses_file: str) -> List[np.ndarray]:
        """
        Carga poses de KITTI desde archivo poses.txt.

        Cada línea contiene 12 valores que representan una matriz de transformación
        3x4 (rotación + traslación) en formato fila. Se reconstruye como matriz
        4x4 homogénea.

        Args:
            poses_file: Ruta al archivo poses.txt

        Returns:
            Lista de matrices 4x4 (una por frame)
        """
        poses = []

        with open(poses_file, 'r') as f:
            for line in f:
                # Parsear 12 valores
                values = [float(x) for x in line.strip().split()]

                if len(values) != 12:
                    raise ValueError(f"Línea de pose inválida: se esperaban 12 valores, se obtuvieron {len(values)}")

                # Convertir a matriz 4x4
                T = np.eye(4, dtype=np.float64)
                T[0, :] = [values[0], values[1], values[2], values[3]]
                T[1, :] = [values[4], values[5], values[6], values[7]]
                T[2, :] = [values[8], values[9], values[10], values[11]]

                poses.append(T)

        return poses

    @staticmethod
    def compute_delta_pose(pose_prev: np.ndarray, pose_current: np.ndarray) -> np.ndarray:
        """
        Calcula la transformación relativa entre dos poses consecutivas.

        En KITTI, pose[i] transforma de world a camera_i. Para transformar
        puntos del frame t-1 al frame t se calcula:
            p_t = T_t^{-1} @ T_{t-1} @ p_{t-1}

        Args:
            pose_prev: Pose del frame anterior (4x4)
            pose_current: Pose del frame actual (4x4)

        Returns:
            delta_pose: Transformación frame t-1 -> frame t (4x4)
        """
        delta_pose = np.linalg.inv(pose_current) @ pose_prev

        return delta_pose

