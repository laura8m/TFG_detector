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
    - Stage 1: Segmentación de suelo (Patchwork++) + HCD
    - Stage 2: Detección de anomalías delta-r
    - Stage 3: Filtrado por clustering DBSCAN
    """

    # ========================================
    # STAGE 1: Segmentación de suelo + HCD
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

    wall_rejection_slope: float = 0.7  # Umbral nz (normal vertical)
    # Si abs(n[2]) < 0.7 → plano inclinado (>45°) → sospecha de pared

    wall_height_diff_threshold: float = 0.3  # Delta-Z local (30cm)
    # Si variación de altura en vecindad > 30cm → confirmada como pared

    wall_kdtree_radius: float = 0.5  # Radio de vecindad local (m)

    # === Height Coding Descriptor (ERASOR++) ===
    # Bandera de ablation: ¿Activar HCD?
    enable_hcd: bool = True

    # Parámetros HCD
    hcd_z_rel_scale: float = 0.3  # Escala normalización altura relativa

    # ========================================
    # STAGE 2: Detección de anomalías delta-r
    # ========================================

    enable_hcd_fusion: bool = True  # Fusionar HCD con delta-r
    threshold_obs: float = -0.5  # Obstáculo positivo (m)
    threshold_void: float = 0.8  # Void/depresión (m)

    # ========================================
    # STAGE 3: Filtrado por clustering DBSCAN
    # ========================================

    enable_cluster_filtering: bool = True  # Activar/desactivar Stage 3
    cluster_eps: float = 0.5  # DBSCAN epsilon (m) - distancia máxima entre puntos del mismo cluster
    cluster_min_samples: int = 5  # DBSCAN min_samples - densidad mínima para core point
    cluster_min_pts: int = 15  # Puntos mínimos por cluster para ser considerado obstáculo real

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
      1. Segmentación de suelo (Patchwork++ + rechazo de paredes + HCD)
      2. Detección de anomalías delta-r con fusión HCD opcional
      3. Filtrado por clustering DBSCAN (elimina ruido disperso)

    Cada etapa puede activarse/desactivarse mediante PipelineConfig para
    facilitar ablation studies.

    Estado compartido entre etapas:
    - self.local_planes: Planos locales por bin CZM para cálculo de delta-r
    - self.hcd: Height Coding Descriptor por punto ground

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
        self.hcd = None  # Height Coding Descriptor (N_ground,)

        # === INICIALIZAR SUBMÓDULOS ===
        self._init_patchwork()
        self.initialize_czm_params()

        if config.verbose:
            print("[INFO] LidarPipelineSuite inicializado")
            print(f"  - Rechazo de paredes: {'SI' if config.enable_hybrid_wall_rejection else 'NO'}")
            print(f"  - HCD (ERASOR++): {'SI' if config.enable_hcd else 'NO'}")
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

        zone_idx = np.full_like(r, -1, dtype=np.int32)
        ring_idx = np.full_like(r, -1, dtype=np.int32)
        sector_idx = np.full_like(r, -1, dtype=np.int32)

        valid = (r > self.config.min_range) & (r <= self.config.max_range)

        # Zona 0
        mask_z0 = valid & (r < self.min_ranges[1])
        if np.any(mask_z0):
            zone_idx[mask_z0] = 0
            ring_idx[mask_z0] = ((r[mask_z0] - self.min_ranges[0]) / self.ring_sizes[0]).astype(np.int32)
            sector_idx[mask_z0] = (theta[mask_z0] / self.sector_sizes[0]).astype(np.int32)

        # Zona 1
        mask_z1 = valid & (r >= self.min_ranges[1]) & (r < self.min_ranges[2])
        if np.any(mask_z1):
            zone_idx[mask_z1] = 1
            ring_idx[mask_z1] = ((r[mask_z1] - self.min_ranges[1]) / self.ring_sizes[1]).astype(np.int32)
            sector_idx[mask_z1] = (theta[mask_z1] / self.sector_sizes[1]).astype(np.int32)

        # Zona 2
        mask_z2 = valid & (r >= self.min_ranges[2]) & (r < self.min_ranges[3])
        if np.any(mask_z2):
            zone_idx[mask_z2] = 2
            ring_idx[mask_z2] = ((r[mask_z2] - self.min_ranges[2]) / self.ring_sizes[2]).astype(np.int32)
            sector_idx[mask_z2] = (theta[mask_z2] / self.sector_sizes[2]).astype(np.int32)

        # Zona 3
        mask_z3 = valid & (r >= self.min_ranges[3])
        if np.any(mask_z3):
            zone_idx[mask_z3] = 3
            ring_idx[mask_z3] = ((r[mask_z3] - self.min_ranges[3]) / self.ring_sizes[3]).astype(np.int32)
            sector_idx[mask_z3] = (theta[mask_z3] / self.sector_sizes[3]).astype(np.int32)

        # Recortar índices por seguridad
        for z in range(4):
            mask = (zone_idx == z)
            if np.any(mask):
                ring_idx[mask] = np.clip(ring_idx[mask], 0, self.config.num_rings_each_zone[z] - 1)
                sector_idx[mask] = np.clip(sector_idx[mask], 0, self.config.num_sectors_each_zone[z] - 1)

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
    # STAGE 1: SEGMENTACIÓN DE SUELO + HCD
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

        # --- Fase 2: Identificar bins con variación vertical sospechosa ---

        suspect_bins = []
        suspect_bin_masks = {}
        unique_bins = np.unique(bin_id)

        for bid in unique_bins:
            mask = bin_id == bid
            indices_in_bin = np.where(mask)[0]

            if len(indices_in_bin) < min_neighbors:
                continue  # Bin con pocos puntos, saltar

            # Análisis delta-Z del bin completo
            z_bin = z[mask]

            if use_percentiles:
                z_high = np.percentile(z_bin, 95)
                z_low = np.percentile(z_bin, 5)
                delta_z = z_high - z_low
            else:
                delta_z = z_bin.max() - z_bin.min()

            # Si el bin tiene escalón vertical, MARCAR como sospechoso (NO rechazar aún)
            if delta_z > delta_z_threshold:
                suspect_bins.append(bid)
                suspect_bin_masks[bid] = mask

        # --- Fase 3: Refinamiento punto a punto con voxel grid (O(N)) ---

        rejected_mask = np.zeros(len(ground_pts), dtype=bool)

        if len(suspect_bins) > 0:
            # Recoger TODOS los índices de puntos sospechosos
            suspect_indices = np.concatenate([
                np.where(suspect_bin_masks[bid])[0] for bid in suspect_bins
            ])

            if len(suspect_indices) > 0:
                # Voxel grid 2D (XY): celdas de 1.0m (= diámetro del radio KDTree)
                # Cada celda tiene ~20-40 puntos → percentiles robustos sin vecinos
                cell_size = kdtree_radius * 2.0
                vox_x = np.floor(ground_pts[:, 0] / cell_size).astype(np.int64)
                vox_y = np.floor(ground_pts[:, 1] / cell_size).astype(np.int64)
                gz = ground_pts[:, 2]

                # Hash espacial → clave única por celda
                voxel_key = vox_x * 100003 + vox_y

                # Ordenar por clave de voxel para agrupar z values
                sort_idx = np.argsort(voxel_key)
                sorted_keys = voxel_key[sort_idx]
                sorted_z = gz[sort_idx].copy()

                # Límites de cada grupo (voxel)
                change_idx = np.where(np.diff(sorted_keys) != 0)[0] + 1
                group_starts = np.concatenate([[0], change_idx])
                group_ends = np.concatenate([change_idx, [len(sorted_keys)]])
                unique_keys = sorted_keys[group_starts]
                counts = group_ends - group_starts

                # Ordenar z dentro de cada grupo para indexar percentiles
                for i in range(len(group_starts)):
                    s, e = group_starts[i], group_ends[i]
                    sorted_z[s:e] = np.sort(sorted_z[s:e])

                # P5 y P95 por voxel (indexar en z ordenado)
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

        # 2. Construir lookup de planos locales
        local_planes = {}

        if len(centers) > 0:
            for i in range(len(centers)):
                c = centers[i]
                n = normals[i]

                bin_id = self.get_czm_bin_scalar(c[0], c[1])

                # Asegurar que la normal apunte hacia arriba
                if n[2] < 0:
                    n = -n

                # Almacenar plano (solo si es suficientemente horizontal)
                # Planos con nz < 0.7 son verticales (paredes) → no son suelo válido
                if bin_id is not None and n[2] >= self.config.wall_rejection_slope:
                    d = -np.dot(n, c)
                    local_planes[bin_id] = (n, d)

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

    def compute_height_coding_descriptor(
        self,
        points: np.ndarray,
        ground_indices: np.ndarray
    ) -> np.ndarray:
        """
        Calcula el Height Coding Descriptor (HCD) basado en ERASOR++.

        El HCD codifica la geometría vertical de cada punto ground midiendo
        su altura relativa al plano local estimado. Permite distinguir entre
        superficies planas (rampas suaves), bordes verticales (bordillos) y
        terreno rugoso.

        Interpretación del descriptor:
        - HCD alto (z_rel alto) → estructura vertical (bordillo, escalón)
        - HCD bajo (z_rel bajo) → rampa suave, terreno plano
        - HCD negativo → punto por debajo del plano local (depresión)

        Versión simplificada vectorizada que solo usa z_rel normalizado con tanh
        para evitar el coste O(N^2) de queries KDTree por punto.

        Args:
            points: (N, 3) todos los puntos
            ground_indices: (M,) índices de puntos ground

        Returns:
            hcd: (M,) descriptor por punto ground en rango [-1, 1]
        """
        if not self.config.enable_hcd:
            return np.zeros(len(ground_indices))

        ground_pts = points[ground_indices]
        hcd = np.zeros(len(ground_pts))

        # 1. Obtener bins de todos los puntos ground (vectorizado)
        z_idx, r_idx, s_idx = self.get_czm_bin(ground_pts[:, 0], ground_pts[:, 1])

        # 2. Crear tabla de lookup de planos
        planes_table_n = np.zeros((4, 4, 54, 3), dtype=np.float32)
        planes_table_d = np.zeros((4, 4, 54), dtype=np.float32)

        # Rellenar con plano global por defecto
        planes_table_n[:, :, :, 2] = 1.0  # nz = 1
        planes_table_d[:, :, :] = self.config.sensor_height

        # Rellenar con planos locales
        for bin_id, (n_loc, d_loc) in self.local_planes.items():
            z_b, r_b, s_b = bin_id
            if 0 <= z_b < 4 and 0 <= r_b < 4 and 0 <= s_b < 54:
                planes_table_n[z_b, r_b, s_b] = n_loc
                planes_table_d[z_b, r_b, s_b] = d_loc

        # 3. Lookup vectorizado de planos
        valid_mask = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)

        z_plane = np.full(len(ground_pts), self.config.sensor_height, dtype=np.float32)

        if np.any(valid_mask):
            valid_idx = np.where(valid_mask)[0]
            z_v = z_idx[valid_idx]
            r_v = r_idx[valid_idx]
            s_v = s_idx[valid_idx]

            n_local = planes_table_n[z_v, r_v, s_v]  # (K, 3)
            d_local = planes_table_d[z_v, r_v, s_v]  # (K,)

            # Calcular z_plane para cada punto: z_plane = -(n.x * x + n.y * y + d) / n.z
            pts_valid = ground_pts[valid_idx]
            nz = n_local[:, 2]
            nz_safe = np.where(np.abs(nz) > 1e-3, nz, 1.0)

            z_plane[valid_idx] = -(
                n_local[:, 0] * pts_valid[:, 0] +
                n_local[:, 1] * pts_valid[:, 1] +
                d_local
            ) / nz_safe

        # 4. Calcular z_rel (altura relativa al plano local)
        z_rel = ground_pts[:, 2] - z_plane

        # 5. HCD simplificado (z_rel normalizado con tanh)
        hcd = np.tanh(z_rel / self.config.hcd_z_rel_scale)

        if self.config.verbose:
            print(f"[Stage 1 HCD] Media: {np.mean(hcd):.3f} +/- {np.std(hcd):.3f}")

        return hcd

    def stage1_complete(self, points: np.ndarray) -> Dict:
        """
        Ejecuta Stage 1 completo: segmentación de suelo + rechazo de paredes + HCD.

        Orquesta las tres sub-etapas de Stage 1:
        1. Patchwork++ para segmentación base ground/nonground
        2. Rechazo híbrido de paredes punto a punto (si habilitado)
        3. Cálculo del Height Coding Descriptor (si habilitado)

        Las paredes rechazadas se reclasifican como nonground (son obstáculos).

        Args:
            points: (N, 3) array de puntos XYZ

        Returns:
            dict con:
            - ground_indices: (M,) índices ground limpios (sin paredes)
            - nonground_indices: (K,) índices nonground (incluyendo paredes)
            - local_planes: Dict de planos locales por bin CZM
            - hcd: (M,) Height Coding Descriptor
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

        # Filtrar ground removiendo wall points (vectorizado)
        # Las paredes pasan a nonground (son obstáculos, no desaparecen)
        if len(rejected_wall_indices) > 0:
            rejected_mask = np.isin(ground_indices, rejected_wall_indices)
            clean_ground = ground_indices[~rejected_mask]
            nonground_indices = np.concatenate([nonground_indices, rejected_wall_indices])
        else:
            clean_ground = ground_indices

        # 3. Height Coding Descriptor
        hcd = self.compute_height_coding_descriptor(points, clean_ground)

        # Almacenar estado
        self.hcd = hcd
        self.rejected_wall_indices = rejected_wall_indices

        timing = (time.time() - t0) * 1000  # ms

        if self.config.verbose:
            print(f"[Stage 1 Completo] {timing:.1f} ms")
            print(f"  Ground: {len(clean_ground)} | Paredes: {len(rejected_wall_indices)}")

        return {
            'ground_indices': clean_ground,
            'nonground_indices': nonground_indices,
            'local_planes': self.local_planes,
            'hcd': hcd,
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
        hcd: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Stage 2: Detección de anomalías delta-r con fusión HCD opcional.

        Mide la desviación entre el rango medido y el rango esperado por el
        plano local:
            delta_r = r_medido - r_esperado

        Donde r_esperado se calcula proyectando el rayo sobre el plano local:
            r_esperado = -d / (n . dirección_rayo)

        Clasificación base:
            delta_r < umbral_obs → Obstáculo positivo (más cerca que el plano)
            delta_r > umbral_void → Void/depresión (más lejos que el plano)
            Intermedio → Ground normal

        Si HCD está disponible, modula la likelihood según la geometría vertical:
        obstáculos con HCD alto reciben mayor confianza, ground plano recibe
        mayor supresión.

        Args:
            points: (N, 3) todos los puntos
            ground_indices: (M,) índices de puntos ground
            n_per_point: (N, 3) normales por punto
            d_per_point: (N,) offsets del plano por punto
            hcd: (M,) Height Coding Descriptor (opcional)

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
        # 1. CALCULAR r_medido (rango real)
        # ========================================
        r_measured = np.sqrt(np.sum(points**2, axis=1))

        # ========================================
        # 2. CALCULAR r_esperado (rango esperado del plano)
        # ========================================
        # Proyectar rayo sobre plano local: r_exp = -d / (n . dir_rayo)
        ray_dir = points / r_measured[:, np.newaxis]  # (N, 3) normalizado
        dot_prod = np.sum(ray_dir * n_per_point, axis=1)  # (N,)

        # Evitar división por cero (rayos paralelos al plano)
        valid_dot = dot_prod < -1e-3  # Solo proyecciones válidas (hacia abajo)

        # Inicializar r_esperado con valor grande (cielo/inválido)
        r_expected = np.full(N, 999.9, dtype=np.float32)

        # Calcular r_esperado solo para proyecciones válidas
        r_expected[valid_dot] = -d_per_point[valid_dot] / dot_prod[valid_dot]

        # ========================================
        # 3. CALCULAR delta_r
        # ========================================
        delta_r = r_measured - r_expected

        # Recortar extremos (evitar outliers numéricos)
        delta_r = np.clip(delta_r, -20.0, 10.0)

        # ========================================
        # 4. CLASIFICACIÓN BASE (sin HCD)
        # ========================================
        threshold_obs = self.config.threshold_obs
        threshold_void = self.config.threshold_void

        # Máscaras base (para likelihood inicial)
        obs_region = delta_r < threshold_obs
        void_region = delta_r > threshold_void
        ground_region = (~obs_region) & (~void_region)

        # Likelihood base (log-odds)
        likelihood_base = np.zeros(N, dtype=np.float32)
        likelihood_base[obs_region] = +2.0   # Obstáculo
        likelihood_base[void_region] = +1.5  # Void
        likelihood_base[ground_region] = -2.0  # Ground

        # ========================================
        # 5. FUSIÓN HCD (opcional)
        # ========================================
        if hcd is not None and self.config.enable_hcd:
            # HCD solo disponible para ground points
            # Crear array completo con HCD=0 para non-ground
            hcd_full = np.zeros(N, dtype=np.float32)
            hcd_full[ground_indices] = hcd

            # Modular likelihood según HCD (vectorizado con np.where)
            likelihood_hcd = likelihood_base.copy()

            # CASO 1: Obstáculos - HCD alto→+4.0, medio→+3.0, bajo→+2.0
            obs_h = hcd_full[obs_region]
            likelihood_hcd[obs_region] = np.where(
                obs_h > 0.5, 4.0, np.where(obs_h > 0.2, 3.0, 2.0))

            # CASO 2: Voids - HCD negativo→+3.0, neutro→+1.5
            void_h = hcd_full[void_region]
            likelihood_hcd[void_region] = np.where(void_h < -0.3, 3.0, 1.5)

            # CASO 3: Ground - HCD plano→-2.5, con textura→-1.5
            gnd_h = hcd_full[ground_region]
            likelihood_hcd[ground_region] = np.where(
                np.abs(gnd_h) < 0.2, -2.5, -1.5)

            likelihood_final = likelihood_hcd

            if self.config.verbose:
                n_boosted = np.sum(likelihood_hcd > likelihood_base)
                print(f"[Stage 2 Fusión HCD] {n_boosted} puntos con likelihood aumentada")

        else:
            likelihood_final = likelihood_base

        # ========================================
        # 6. GENERAR MÁSCARAS FINALES USANDO LIKELIHOOD
        # ========================================
        # Umbral de likelihood para clasificar como obstáculo
        likelihood_threshold_obs = 1.0
        likelihood_threshold_gnd = -1.0

        obs_mask_final = likelihood_final > likelihood_threshold_obs
        ground_mask_final = likelihood_final < likelihood_threshold_gnd
        void_mask_final = void_region  # Mantener void mask original

        # Forzar paredes rechazadas en Stage 1 como obstáculos
        # Evita que delta_r~0 (plano vertical) las reclasifique como ground
        if hasattr(self, 'rejected_wall_indices') and len(self.rejected_wall_indices) > 0:
            wall_mask = np.zeros(N, dtype=bool)
            valid_idx = self.rejected_wall_indices[self.rejected_wall_indices < N]
            wall_mask[valid_idx] = True
            obs_mask_final[wall_mask] = True
            ground_mask_final[wall_mask] = False

        uncertain_mask = (~obs_mask_final) & (~ground_mask_final) & (~void_mask_final)

        # ========================================
        # 7. MÉTRICAS Y RETORNO
        # ========================================
        n_obs = np.sum(obs_mask_final)
        n_void = np.sum(void_mask_final)
        n_ground = np.sum(ground_mask_final)
        n_uncertain = np.sum(uncertain_mask)

        t_end = time.time()
        timing_ms = (t_end - t_start) * 1000.0

        if self.config.verbose:
            hcd_status = "SI" if (hcd is not None and self.config.enable_hcd) else "NO"
            print(f"[Stage 2 Completo] {timing_ms:.1f} ms")
            print(f"  Obstáculos: {n_obs} | Voids: {n_void} | Ground: {n_ground} | Inciertos: {n_uncertain}")
            print(f"  Fusión HCD: {hcd_status}")
            if hcd is not None and self.config.enable_hcd:
                n_hcd_changed = np.sum(obs_mask_final != obs_region)
                print(f"  HCD cambió {n_hcd_changed} clasificaciones")

        return {
            'delta_r': delta_r,
            'likelihood': likelihood_final,
            'obs_mask': obs_mask_final,
            'void_mask': void_mask_final,
            'ground_mask': ground_mask_final,
            'uncertain_mask': uncertain_mask,
            'timing_ms': timing_ms
        }

    def stage2_complete(self, points: np.ndarray) -> Dict:
        """
        Stage 2 completo: ejecuta Stage 1 + delta-r con fusión HCD.

        Orquesta la ejecución secuencial de Stage 1 (segmentación) y
        Stage 2 (detección de anomalías), reutilizando las normales y
        distancias de plano calculadas en Stage 1.

        Args:
            points: (N, 3) nube de puntos

        Returns:
            Dict con resultados combinados de Stage 1 + Stage 2
        """
        # Stage 1: Segmentación de suelo + HCD
        stage1_result = self.stage1_complete(points)

        # Stage 2: delta-r con fusión HCD
        stage2_result = self.compute_delta_r(
            points=points,
            ground_indices=stage1_result['ground_indices'],
            n_per_point=stage1_result['n_per_point'],
            d_per_point=stage1_result['d_per_point'],
            hcd=stage1_result['hcd'] if self.config.enable_hcd else None
        )

        # Combinar resultados
        return {
            **stage1_result,  # ground_indices, rejected_walls, hcd, timing_ms (Stage 1)
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
        # 2. CLUSTERING DBSCAN
        # ========================================
        t_dbscan = time.time()

        db = DBSCAN(
            eps=cfg.cluster_eps,
            min_samples=cfg.cluster_min_samples,
            n_jobs=-1  # Usar todos los cores
        )
        cluster_labels_obs = db.fit_predict(obs_pts)  # (M,) etiquetas: -1=ruido, 0,1,2...=cluster

        t_dbscan_end = time.time()

        # ========================================
        # 3. FILTRAR CLUSTERS PEQUEÑOS
        # ========================================
        t_filter = time.time()

        # Contar puntos por cluster
        unique_labels = np.unique(cluster_labels_obs)
        valid_clusters = set()

        for label in unique_labels:
            if label == -1:  # Ruido
                continue
            cluster_size = (cluster_labels_obs == label).sum()
            if cluster_size >= cfg.cluster_min_pts:
                valid_clusters.add(label)

        # Máscara: punto pertenece a cluster válido (grande)
        valid_mask_obs = np.array([
            cl in valid_clusters for cl in cluster_labels_obs
        ], dtype=bool)

        # Actualizar obs_mask: solo mantener puntos en clusters válidos
        obs_mask_new = np.zeros(N, dtype=bool)
        obs_mask_new[obs_indices[valid_mask_obs]] = True

        # Propagar al array completo de etiquetas de cluster
        cluster_labels_full = np.full(N, -1, dtype=np.int32)
        cluster_labels_full[obs_indices] = cluster_labels_obs

        t_filter_end = time.time()

        # ========================================
        # 4. MÉTRICAS
        # ========================================
        t_end = time.time()
        timing_ms = (t_end - t_start) * 1000.0

        n_clusters_total = len(unique_labels[unique_labels >= 0])
        n_clusters_valid = len(valid_clusters)
        n_clusters_rejected = n_clusters_total - n_clusters_valid
        n_noise = (cluster_labels_obs == -1).sum()
        n_small_cluster = (~valid_mask_obs & (cluster_labels_obs >= 0)).sum()
        n_removed = int(obs_mask.sum() - obs_mask_new.sum())

        if cfg.verbose:
            print(f"[Stage 3 DBSCAN] {timing_ms:.1f} ms")
            print(f"  Clusters: {n_clusters_total} total | {n_clusters_valid} válidos (>={cfg.cluster_min_pts} pts) | {n_clusters_rejected} rechazados")
            print(f"  Puntos eliminados: {n_removed} ({100*n_removed/max(n_obs,1):.1f}%) — ruido: {n_noise}, clusters pequeños: {n_small_cluster}")
            print(f"  Tiempos: DBSCAN={1000*(t_dbscan_end-t_dbscan):.0f}ms | filtro={1000*(t_filter_end-t_filter):.0f}ms")

        return {
            **stage2_result,
            'obs_mask': obs_mask_new,
            'cluster_labels': cluster_labels_full,
            'n_clusters': n_clusters_valid,
            'n_clusters_rejected': n_clusters_rejected,
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

