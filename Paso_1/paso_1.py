#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import argparse
import sys
from pathlib import Path
import cv2
import struct
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
import colorsys
from scipy.spatial import Delaunay, cKDTree
import math
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time as _time
import os
from tf2_ros import StaticTransformBroadcaster

# Añadir directorio actual al path para encontrar modulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smoothing_utils import smooth_chaikin

# Añadir Patchwork++ al path (desde el entorno virtual pwenv)

class Step1AnomalyDetector(Node):
    """
    Paso 1: Nodo para detectar anomalías geométricas sobre el terreno.
    Realiza segmentación de suelo y detecta obstáculos iniciales usando proyeccion de rango.
    """
    def __init__(self):
        super().__init__('step1_anomaly_detector')
        
        # --- PARÁMETROS ---
        # Fixed path for Step 1
        self.bin_file_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/data/000000.bin")
        self.scene = "00"
        
        # Parámetros del Sensor (Velodyne HDL-64E)
        self.H = 64 # Altura de imagen (número de anillos del sensor)
        self.W = 2048 # Anchura de imagen (resolución horizontal)
        self.fov_up = 3.0 * np.pi / 180.0 # Campo de visión superior (+3 grados)
        self.fov_down = -25.0 * np.pi / 180.0 # Campo de visión inferior (-25 grados)
        self.fov = abs(self.fov_down) + abs(self.fov_up) # Rango vertical total (~28 grados)
        self.min_range = 2.7 # Distancia mínima (filtrar ego-vehículo)
        self.max_range = 80.0 # Distancia máxima efectiva
        
        # Parámetros de Segmentación de Suelo
        self.ground_z_threshold = -1.5 
        self.default_ground_height = 1.73
        self.wall_rejection_slope = 0.7
        self.wall_height_diff_threshold = 0.3
        
        # Parámetros Iniciales de Bayes (usados para los deltas)
        self.l0 = 0.0
        self.threshold_obs = -0.3
        self.belief_clamp_min = -5.0
        self.belief_clamp_max = 5.0
        self.prob_threshold_obs = 0.6
        self.prob_threshold_gnd = 0.4
        
        # Calibración
        self.Tr = np.array([
            [4.2768028e-04, -9.9996725e-01, -8.0844917e-03, -1.1984599e-02],
            [-7.2106265e-03, 8.0811985e-03, -9.9994132e-01, -5.4039847e-02],
            [9.9997386e-01, 4.8594858e-04, -7.2069002e-03, -2.9219686e-01],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Estado Inicial
        self.current_scan = 0
        self.frame_count = 0
        self.belief_map = np.zeros((self.H, self.W), dtype=np.float64)
        self.current_pose = np.eye(4)
        
        # Modulos de ROS 2
        self.cv_bridge = CvBridge()
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()
        self.init_publishers()
        
        # Inicializar Patchwork++
        self.init_patchwork()
        
        # Ejecutar Frame Único Inmediatamente
        self.process_frame()
        self.get_logger().info("Paso 1: Anomalías geométricas detectadas y publicadas.")
        
        # Timer para republicar visualizaciones (loop)
        self.timer = self.create_timer(1.0, self.publish_visualization)

    def init_publishers(self):
        self.point_cloud_pub = self.create_publisher(PointCloud2, 'point_cloud_raw', 10)
        self.range_image_pub = self.create_publisher(Image, 'range_image_raw', 10)
        self.delta_r_cloud_pub = self.create_publisher(PointCloud2, 'delta_r_cloud', 10)
        self.ground_cloud_pub = self.create_publisher(PointCloud2, 'ground_cloud_raw', 10)

    def broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'velodyne'
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def init_patchwork(self):
        try:
            import pypatchworkpp
            self.params = pypatchworkpp.Parameters()
            self.params.verbose = False
            self.params.sensor_height = self.default_ground_height
            self.params.min_range = self.min_range
            self.params.max_range = self.max_range
            self.params.num_iter = 3
            self.params.num_lpr = 20
            self.params.num_min_pts = 10
            self.params.th_dist = 0.2
            self.params.uprightness_thr = 0.707
            self.params.adaptive_seed_selection_margin = -1.1
            self.params.enable_RNR = False
            
            # CZM
            self.params.num_zones = 4
            self.params.num_rings_each_zone = [2, 4, 4, 4]
            self.params.num_sectors_each_zone = [16, 32, 54, 32]

            self.patchwork = pypatchworkpp.patchworkpp(self.params)
            self.initialize_czm_params()
        except ImportError:
            self.get_logger().error("No se pudo importar pypatchworkpp. Asegúrate de compilar e incluir la ruta en sys.path")
            sys.exit(1)

    def initialize_czm_params(self):
        min_r = self.params.min_range
        max_r = self.params.max_range
        self.min_ranges = [min_r, (7*min_r+max_r)/8.0, (3*min_r+max_r)/4.0, (min_r+max_r)/2.0]
        self.ring_sizes = []
        for i in range(4):
            end_r = self.min_ranges[i+1] if i < 3 else max_r
            self.ring_sizes.append((end_r - self.min_ranges[i]) / self.params.num_rings_each_zone[i])
        self.sector_sizes = [2 * np.pi / n for n in self.params.num_sectors_each_zone]

    def process_frame(self):
        t_start = _time.time()
        
        # 1. Cargar Datos (Ejemplo estático)
        if not self.bin_file_path.exists():
            self.get_logger().error(f"Archivo no encontrado: {self.bin_file_path}")
            return
            
        scan = np.fromfile(self.bin_file_path, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        
        # Publicar Nube Original
        self.raw_cloud_msg = self.create_cloud(points, np.ones(len(points))*255.0, 'intensity')
        
        t1 = _time.time()

        # 2. Segmentación de Suelo
        ground_points, n_per_point, d_per_point, rejected_mask = self.segment_ground(points)
        self.rejected_mask = rejected_mask 
        
        t2 = _time.time()
        
        # 3. Proyección de Rango & Detección Básica (Delta R)
        range_image, delta_r, u, v, r, r_exp = self.compute_range_projection(points, n_per_point, d_per_point)
        
        t3 = _time.time()
        
        # 4. Generar Mensajes de Detección
        self.generate_step1_visualizations(points, range_image, delta_r, ground_points)
        
        t_end = _time.time()
        
        self.get_logger().info(f"--- TIEMPOS ---")
        self.get_logger().info(f"Carga Datos:      {(t1 - t_start) * 1000:.1f} ms")
        self.get_logger().info(f"Suelo Patchwork+: {(t2 - t1) * 1000:.1f} ms")
        self.get_logger().info(f"Proyeccion Rango: {(t3 - t2) * 1000:.1f} ms")
        self.get_logger().info(f"RViz Msg Gen:     {(t_end - t3) * 1000:.1f} ms")
        self.get_logger().info(f"TOTAL:            {(t_end - t_start) * 1000:.1f} ms")

    def segment_ground(self, points):
        """
        Segmentación de suelo usando Patchwork++ con filtrado de paredes.

        Returns:
            ground_points: Puntos clasificados como suelo por Patchwork++
            n_per_point: Normal del plano asignado a cada punto
            d_per_point: Parámetro d del plano asignado a cada punto
            rejected_mask: Máscara de puntos en bins rechazados (paredes)
        """
        self.patchwork.estimateGround(points)
        ground_points = self.patchwork.getGround()

        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()

        # Filtrar planos verticales (paredes) antes de usarlos
        local_planes, rejected_bins = self.filter_wall_planes(centers, normals)

        global_n, global_d = self.compute_global_plane(ground_points)
        n_per_point, d_per_point = self.assign_planes_to_points(points, local_planes, global_n, global_d)

        # Crear máscara de bins rechazados (puntos en bins de paredes)
        rejected_mask = self.create_rejected_mask(points, rejected_bins)

        return ground_points, n_per_point, d_per_point, rejected_mask

    def filter_wall_planes(self, centers, normals):
        """
        Filtra planos verticales (paredes) basándose en la componente vertical de la normal.

        Args:
            centers: Centroides de los planos generados por Patchwork++
            normals: Vectores normales de los planos

        Returns:
            local_planes: Dict {bin_id: (normal, d)} solo con planos horizontales (suelo)
            rejected_bins: Set de bin_ids rechazados (paredes)
        """
        local_planes = {}
        rejected_bins = set()
        n_rejected = 0
        n_accepted = 0

        for c, n in zip(centers, normals):
            bin_id = self.get_czm_bin_scalar(c[0], c[1])
            if not bin_id:
                continue

            # Asegurar normal apuntando hacia arriba
            if n[2] < 0:
                n = -n

            # CRITERIO DE RECHAZO: Normal horizontal (nz < threshold)
            # nz ≈ 1.0 → plano horizontal (suelo)
            # nz ≈ 0.0 → plano vertical (pared)
            # threshold = 0.7 → cos(45°) → rechaza planos con inclinación > 45°

            if abs(n[2]) < self.wall_rejection_slope:
                # Plano vertical -> RECHAZAR (es una pared)
                n_rejected += 1
                rejected_bins.add(bin_id)
                # NO agregar a local_planes
                # Los puntos en este bin usarán el plano global (fallback)
                continue

            # Plano horizontal -> ACEPTAR (es suelo)
            n_accepted += 1
            local_planes[bin_id] = (n, -np.dot(n, c))

        self.get_logger().info(
            f"Plane filtering: {n_accepted} ground planes accepted, "
            f"{n_rejected} wall planes rejected ({len(rejected_bins)} bins)"
        )

        return local_planes, rejected_bins

    def create_rejected_mask(self, points, rejected_bins):
        """
        Crea máscara booleana de puntos que pertenecen a bins rechazados (paredes).

        Args:
            points: Nube de puntos (N x 3)
            rejected_bins: Set de bin_ids rechazados

        Returns:
            rejected_mask: Array booleano (N,) True si el punto está en un bin de pared
        """
        if len(rejected_bins) == 0:
            return np.zeros(len(points), dtype=bool)

        # Obtener índices CZM de todos los puntos
        z_idx, r_idx, s_idx = self.get_czm_bin_vectorized(points[:,0], points[:,1])

        # Inicializar máscara
        rejected_mask = np.zeros(len(points), dtype=bool)

        # Marcar puntos en bins rechazados
        for z_bin, r_bin, s_bin in rejected_bins:
            mask = (z_idx == z_bin) & (r_idx == r_bin) & (s_idx == s_bin)
            rejected_mask |= mask

        n_rejected_points = np.sum(rejected_mask)
        self.get_logger().info(f"Points in wall bins: {n_rejected_points} ({100*n_rejected_points/len(points):.1f}%)")

        return rejected_mask

    def compute_global_plane(self, ground_points):
        if len(ground_points) > 10:
            centroid = np.mean(ground_points, axis=0)
            u, s, vt = np.linalg.svd(ground_points - centroid, full_matrices=False)
            normal = vt[2, :]
            if normal[2] < 0: normal = -normal
            d = -np.dot(normal, centroid)
            return normal, d
        return np.array([0,0,1]), 1.73

    def assign_planes_to_points(self, points, local_planes, global_n, global_d):
        z_idx, r_idx, s_idx = self.get_czm_bin_vectorized(points[:,0], points[:,1])
        
        # Valid mask
        valid = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)
        
        n_out = np.full((len(points), 3), global_n, dtype=np.float32)
        d_out = np.full(len(points), global_d, dtype=np.float32)
        
        if not np.any(valid):
            return n_out, d_out
            
        # Optimization: Map valid indices directly without creating a 4x4x54x4 table
        max_idx = 4 * 4 * 54
        flat_n = np.full((max_idx, 3), global_n, dtype=np.float32)
        flat_d = np.full(max_idx, global_d, dtype=np.float32)
        
        for (z, r, s), (n, d) in local_planes.items():
            if 0<=z<4 and 0<=r<4 and 0<=s<54:
                idx = z * (4 * 54) + r * 54 + s
                flat_n[idx] = n
                flat_d[idx] = d
                
        # Apply flat mapping vectorized
        flat_indices = z_idx[valid] * (4 * 54) + r_idx[valid] * 54 + s_idx[valid]
        n_out[valid] = flat_n[flat_indices]
        d_out[valid] = flat_d[flat_indices]
            
        return n_out, d_out

    def get_czm_bin_scalar(self, x, y):
        r = np.sqrt(x*x + y*y)
        if r <= self.params.min_range or r > self.params.max_range: return None
        theta = np.arctan2(y, x)
        if theta < 0: theta += 2*np.pi
        
        for z in range(4):
            r_end = self.min_ranges[z+1] if z < 3 else self.params.max_range
            if r < r_end:
                r_start = self.min_ranges[z]
                r_idx = int((r - r_start) / self.ring_sizes[z])
                r_idx = min(r_idx, self.params.num_rings_each_zone[z]-1)
                s_idx = int(theta / self.sector_sizes[z])
                s_idx = min(s_idx, self.params.num_sectors_each_zone[z]-1)
                return (z, r_idx, s_idx)
        return None

    def get_czm_bin_vectorized(self, x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2*np.pi
        
        valid = (r > self.params.min_range) & (r <= self.params.max_range)
        
        z_idx = np.full(len(r), -1, dtype=np.int32)
        r_idx = np.full(len(r), -1, dtype=np.int32)
        s_idx = np.full(len(r), -1, dtype=np.int32)
        
        if not np.any(valid):
            return z_idx, r_idx, s_idx
            
        # Use digitize to find the zone (z) efficiently
        # np.digitize returns 1 for values in first bin, we want 0-based
        z_assigned = np.digitize(r[valid], self.min_ranges) - 1
        # Fix max_range edge case where digitize might put it in bin 4
        z_assigned = np.clip(z_assigned, 0, 3)
        
        z_idx[valid] = z_assigned
        
        # Calculate r_idx and s_idx vectorially based on the assigned zone
        ring_sizes_arr = np.array(self.ring_sizes)[z_assigned]
        min_ranges_arr = np.array(self.min_ranges[:4])[z_assigned]
        num_rings_arr = np.array(self.params.num_rings_each_zone)[z_assigned]
        
        sector_sizes_arr = np.array(self.sector_sizes)[z_assigned]
        num_sectors_arr = np.array(self.params.num_sectors_each_zone)[z_assigned]
        
        # r_idx calculation
        r_calc = ((r[valid] - min_ranges_arr) / ring_sizes_arr).astype(np.int32)
        # Fast element-wise min/max clipping
        r_idx[valid] = np.minimum(np.maximum(r_calc, 0), num_rings_arr - 1)
        
        # s_idx calculation
        s_calc = (theta[valid] / sector_sizes_arr).astype(np.int32)
        s_idx[valid] = np.minimum(np.maximum(s_calc, 0), num_sectors_arr - 1)
                
        return z_idx, r_idx, s_idx

    def compute_range_projection(self, points, n, d):
        r = np.linalg.norm(points, axis=1)
        dot = np.sum(points * n, axis=1)
        valid_dot = dot < -1e-3
        
        r_exp = np.full(len(points), 999.9, dtype=np.float32)
        r_exp[valid_dot] = -d[valid_dot] * r[valid_dot] / dot[valid_dot]
        
        delta_r = r - r_exp
        # Clip delta bounds for simpler logic
        delta_r = np.clip(delta_r, -20.0, 10.0)
        
        u, v, valid_fov = self.project_points_to_uv(points, r)
        
        range_image = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Z-Buffer invertido (Closest overrides far)
        order = np.argsort(r)[::-1]
        u_s, v_s, d_s = u[order], v[order], delta_r[order]
        range_image[u_s, v_s] = d_s
        
        return range_image, delta_r, u, v, r, r_exp

    def project_points_to_uv(self, points, r=None):
        if r is None: r = np.linalg.norm(points, axis=1)
        valid = r > 0.1
        
        # Optimization: use out/where directly to avoid intermediate array copies
        z_r = np.zeros_like(r)
        np.divide(points[:,2], r, out=z_r, where=valid)
        z_r = np.clip(z_r, -1.0, 1.0)
        
        pitch = np.arcsin(z_r)
        yaw = np.arctan2(points[:,1], points[:,0])
        
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        # Fast floor and cast
        u = (proj_y * self.H).astype(np.int32)
        u = np.clip(u, 0, self.H - 1)
        
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        v = (proj_x * self.W).astype(np.int32)
        v = np.clip(v, 0, self.W - 1)
        
        return u, v, valid

    def generate_step1_visualizations(self, points, range_image, delta_r, ground):
        mask_obs = delta_r < self.threshold_obs
        mask_dep = (delta_r > 0.2) & (delta_r < 2.0)
        mask_sky = delta_r >= 2.0
        
        mask_dep_clean = mask_dep & ~mask_obs
        mask_sky_clean = mask_sky & ~mask_obs
        
        # Vectorized color assignment
        rgb = np.full(len(points), 0x808080, dtype=np.uint32) # Default gray (128,128,128)
        rgb[mask_obs] = 0x0000FF # Blue
        rgb[mask_dep_clean] = 0xFFA500 # Orange
        rgb[mask_sky_clean] = 0x00FFFF # Cyan
        
        self.delta_r_msg = self.create_cloud(points, rgb.view(np.float32), 'rgb')
            
        self.ground_cloud_msg = self.create_cloud(ground, np.zeros(len(ground), dtype=np.float32), 'intensity')
            
        # Avoid re-allocating if min == max
        r_min, r_max = range_image.min(), range_image.max()
        if r_max > r_min:
            cv_img = ((range_image - r_min) / (r_max - r_min) * 255.0).astype(np.uint8)
        else:
            cv_img = np.zeros_like(range_image, dtype=np.uint8)
        self.range_image_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="mono8")
        self.range_image_msg.header.frame_id = "velodyne"

    def create_cloud(self, points, values, field_name):
        msg = PointCloud2()
        msg.header.frame_id = "velodyne"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name=field_name, offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(points)
        msg.is_dense = True
        
        # Fast structured array creation
        arr = np.empty(len(points), dtype=[('x','f4'),('y','f4'),('z','f4'),(field_name,'f4')])
        arr['x'] = points[:,0]
        arr['y'] = points[:,1]
        arr['z'] = points[:,2]
        arr[field_name] = values
        
        msg.data = arr.tobytes()
        return msg

    def publish_visualization(self):
        t = self.get_clock().now().to_msg()
        if hasattr(self, 'delta_r_msg'):
            self.delta_r_msg.header.stamp = t
            self.delta_r_cloud_pub.publish(self.delta_r_msg)
        if hasattr(self, 'ground_cloud_msg'):
            self.ground_cloud_msg.header.stamp = t
            self.ground_cloud_pub.publish(self.ground_cloud_msg)
        if hasattr(self, 'range_image_msg'):
            self.range_image_msg.header.stamp = t
            self.range_image_pub.publish(self.range_image_msg)
        if hasattr(self, 'raw_cloud_msg'):
            self.raw_cloud_msg.header.stamp = t
            self.point_cloud_pub.publish(self.raw_cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Step1AnomalyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
