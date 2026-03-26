#!/usr/bin/env python3
"""
Visualizador del pipeline LiDAR por stages en RViz.

Publica la nube de puntos coloreada por clasificación después de cada stage,
para ver la evolución del pipeline.

Pipeline de 2 stages:
  1. Ground estimation (Patchwork++ + wall rejection)
  2. Delta-r conservador (opcional, --delta_r)

Colores en RViz (usar PointCloud2 con color por campo 'rgb'):
  - Verde:    suelo (ground)
  - Rojo:     obstáculo
  - Azul:     void/depresión
  - Amarillo: pared rechazada
  - Gris:     incierto/sin clasificar

Uso:
  python3 run_pipeline_viz.py --seq 00 --scan 50
  python3 run_pipeline_viz.py --seq 04 --scan_start 0 --scan_end 10
  python3 run_pipeline_viz.py --seq 00 --scan 50 --stages 1 2 3
"""

import numpy as np
import argparse
import sys
import os
import struct
import time
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from pathlib import Path

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig
from data_paths import get_sequence_info, get_scan_file, get_label_file


# ========================================
# DATOS KITTI
# ========================================

def load_scan(scan_id, seq, curb_labels=False):
    """Siempre carga bin completo de SemanticKITTI. curb_labels ya no afecta al bin."""
    scan_file = get_scan_file(seq, scan_id)
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]
    return points


def load_gt_labels(scan_id, seq):
    label_file = get_label_file(seq, scan_id)
    if label_file.exists():
        labels = np.fromfile(label_file, dtype=np.uint32)
        return labels & 0xFFFF
    return None


# ========================================
# COLORES
# ========================================

def rgb_to_float(r, g, b):
    """Convierte RGB (0-255) a float32 para PointCloud2 campo 'rgb'."""
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', rgb_int))[0]

COLOR_GROUND    = rgb_to_float(50, 200, 50)    # Verde
COLOR_OBSTACLE  = rgb_to_float(230, 50, 50)    # Rojo
COLOR_VOID      = rgb_to_float(50, 100, 230)   # Azul
COLOR_WALL      = rgb_to_float(230, 200, 50)   # Amarillo
COLOR_UNCERTAIN = rgb_to_float(150, 150, 150)  # Gris
COLOR_CURB      = rgb_to_float(0, 230, 230)    # Cyan
COLOR_VOID_NEW  = rgb_to_float(255, 140, 0)    # Naranja (hoyos)

# Paleta de colores para clusters DBSCAN
CLUSTER_COLORS = [
    rgb_to_float(230, 25, 75),    # Rojo
    rgb_to_float(60, 180, 75),    # Verde
    rgb_to_float(255, 225, 25),   # Amarillo
    rgb_to_float(0, 130, 200),    # Azul
    rgb_to_float(245, 130, 48),   # Naranja
    rgb_to_float(145, 30, 180),   # Púrpura
    rgb_to_float(70, 240, 240),   # Cyan
    rgb_to_float(240, 50, 230),   # Magenta
    rgb_to_float(210, 245, 60),   # Lima
    rgb_to_float(250, 190, 212),  # Rosa
    rgb_to_float(0, 128, 128),    # Teal
    rgb_to_float(220, 190, 255),  # Lavanda
    rgb_to_float(170, 110, 40),   # Marrón
    rgb_to_float(128, 0, 0),      # Granate
    rgb_to_float(0, 0, 128),      # Navy
    rgb_to_float(128, 128, 0),    # Oliva
]
# Colores por categoría SemanticKITTI para Ground Truth
GT_LABEL_COLORS = {
    # Bordillo (3D-Curb)
    3: rgb_to_float(0, 255, 255),      # curb (cyan)
    # Vehículos - tonos azul/cyan
    10: rgb_to_float(0, 0, 230),      # car (azul)
    11: rgb_to_float(0, 0, 180),      # bicycle (azul oscuro)
    13: rgb_to_float(0, 80, 230),     # bus (azul claro)
    15: rgb_to_float(0, 150, 255),    # motorcycle (cyan-azul)
    16: rgb_to_float(0, 60, 180),     # on-rails
    18: rgb_to_float(0, 100, 230),    # truck (azul medio)
    20: rgb_to_float(0, 40, 140),     # other-vehicle
    # Personas - tonos rojo/naranja
    30: rgb_to_float(255, 30, 30),    # person (rojo)
    31: rgb_to_float(255, 120, 0),    # bicyclist (naranja)
    32: rgb_to_float(255, 80, 40),    # motorcyclist (rojo-naranja)
    # Estructuras - tonos rosa/magenta
    40: rgb_to_float(200, 50, 200),   # road (no se usa como obstáculo)
    44: rgb_to_float(230, 100, 230),  # parking
    48: rgb_to_float(180, 50, 180),   # sidewalk
    49: rgb_to_float(160, 30, 160),   # other-ground
    # Señales/postes - amarillo
    50: rgb_to_float(255, 255, 50),   # building (amarillo)
    51: rgb_to_float(255, 200, 50),   # fence (amarillo oscuro)
    52: rgb_to_float(200, 200, 0),    # other-structure
    # Naturaleza - tonos verdes
    70: rgb_to_float(0, 180, 0),      # vegetation (verde)
    71: rgb_to_float(100, 230, 100),  # trunk (verde claro)
    72: rgb_to_float(150, 200, 100),  # terrain (verde-amarillo)
    80: rgb_to_float(80, 150, 80),    # pole (verde oscuro)
    81: rgb_to_float(120, 180, 60),   # traffic-sign
    # Otros
    99: rgb_to_float(200, 200, 200),  # other-object (gris)
    # Moving objects - tonos más brillantes
    252: rgb_to_float(80, 80, 255),   # moving-car
    253: rgb_to_float(255, 80, 80),   # moving-person
    254: rgb_to_float(255, 160, 60),  # moving-bicyclist
    255: rgb_to_float(255, 130, 80),  # moving-motorcyclist
    256: rgb_to_float(60, 60, 200),   # moving-bus
    257: rgb_to_float(40, 140, 255),  # moving-truck
    258: rgb_to_float(100, 100, 255), # moving-other-vehicle
    259: rgb_to_float(80, 80, 220),   # moving-on-rails
}

GT_OBSTACLE_LABELS = set(GT_LABEL_COLORS.keys())


# ========================================
# NODO ROS 2
# ========================================

class PipelineVizNode(Node):
    def __init__(self, args):
        super().__init__('pipeline_viz')
        self.args = args

        # Publishers por stage
        self.pub_vanilla = self.create_publisher(PointCloud2, '/patchwork_vanilla_cloud', 10)
        self.pub_stage1 = self.create_publisher(PointCloud2, '/stage1_cloud', 10)
        self.pub_stage2 = self.create_publisher(PointCloud2, '/stage2_cloud', 10)
        self.pub_clusters = self.create_publisher(PointCloud2, '/cluster_cloud', 10)
        self.pub_gt     = self.create_publisher(PointCloud2, '/gt_cloud', 10)
        self.pub_curb_gt = self.create_publisher(PointCloud2, '/curb_gt_cloud', 10)

        # Timer para ejecutar una vez tras inicializar
        self.timer = self.create_timer(1.0, self.run_pipeline)
        self.done = False

    def create_rgb_cloud(self, points, rgb_values):
        """Crea PointCloud2 con campo RGB para colorear en RViz."""
        msg = PointCloud2()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "velodyne"
        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)]
        cloud = np.empty(len(points), dtype=dtype_list)
        cloud['x'] = points[:, 0].astype(np.float32)
        cloud['y'] = points[:, 1].astype(np.float32)
        cloud['z'] = points[:, 2].astype(np.float32)
        cloud['rgb'] = rgb_values.astype(np.float32)

        msg.data = cloud.tobytes()
        return msg

    def publish_gt(self, scan_id, seq, points):
        """Publica ground truth coloreado por etiqueta."""
        if self.args.curb_labels:
            curb_label_file = Path(os.path.dirname(os.path.abspath(__file__))) / '3d_curb_labels' / seq / 'labels' / f'{scan_id:06d}.label'
            if curb_label_file.exists():
                labels = np.fromfile(str(curb_label_file), dtype=np.uint32) & 0xFFFF
            else:
                labels = load_gt_labels(scan_id, seq)
        else:
            labels = load_gt_labels(scan_id, seq)
        if labels is None:
            self.get_logger().warn("No GT labels found")
            return

        N = min(len(points), len(labels))
        gt_mask = np.isin(labels[:N], list(GT_OBSTACLE_LABELS))
        gt_points = points[:N][gt_mask]
        gt_labels = labels[:N][gt_mask]

        if len(gt_points) > 0:
            rgb = np.full(len(gt_points), rgb_to_float(200, 200, 200), dtype=np.float32)
            for lbl, color in GT_LABEL_COLORS.items():
                mask = (gt_labels == lbl)
                if np.any(mask):
                    rgb[mask] = color
            msg = self.create_rgb_cloud(gt_points, rgb)
            self.pub_gt.publish(msg)
            self.get_logger().info(f"  GT: {len(gt_points)} obstacle points")

    def publish_curb_gt(self, scan_id, seq, points, obs_mask):
        """Publica bordillos GT: cyan=detectado, magenta=perdido."""
        # Buscar labels con curb=3 (mapeados o originales)
        curb_label_file = get_label_file(seq, scan_id, use_curb=True)
        if curb_label_file.exists():
            labels = np.fromfile(str(curb_label_file), dtype=np.uint32) & 0xFFFF
        else:
            # Fallback a labels normales
            labels = load_gt_labels(scan_id, seq)
        if labels is None:
            return
        N = min(len(points), len(labels))
        curb_mask = labels[:N] == 3  # curb label
        if not np.any(curb_mask):
            return

        curb_points = points[:N][curb_mask]
        curb_detected = obs_mask[:N][curb_mask]

        rgb = np.full(len(curb_points), rgb_to_float(255, 0, 255), dtype=np.float32)  # Magenta = perdido
        rgb[curb_detected] = rgb_to_float(0, 255, 255)  # Cyan = detectado

        self.pub_curb_gt.publish(self.create_rgb_cloud(curb_points, rgb))
        n_det = int(curb_detected.sum())
        n_tot = len(curb_points)
        self.get_logger().info(f"  Curb GT: {n_det}/{n_tot} detectados ({100*n_det/n_tot:.1f}%)")

    def colorize_vanilla(self, points):
        """PW++ vanilla: solo ground/non-ground sin WR."""
        N = len(points)
        rgb = np.full(N, COLOR_UNCERTAIN, dtype=np.float32)

        self.pipeline.patchwork.estimateGround(points)
        ground_idx = np.array(self.pipeline.patchwork.getGroundIndices())

        ground_mask = np.zeros(N, dtype=bool)
        if len(ground_idx) > 0:
            valid = ground_idx[ground_idx < N]
            ground_mask[valid] = True

        rgb[ground_mask] = COLOR_GROUND
        rgb[~ground_mask] = COLOR_OBSTACLE

        return rgb

    def colorize_stage1(self, points, result):
        """Stage 1: suelo vs no-suelo + paredes rechazadas."""
        N = len(points)
        rgb = np.full(N, COLOR_UNCERTAIN, dtype=np.float32)

        ground_idx = result['ground_indices']
        nonground_idx = result['nonground_indices']
        walls = result.get('rejected_walls', np.array([], dtype=int))

        rgb[ground_idx] = COLOR_GROUND
        rgb[nonground_idx] = COLOR_OBSTACLE

        # Paredes rechazadas (son índices dentro de ground original)
        if len(walls) > 0 and len(ground_idx) > 0:
            # Las paredes fueron rechazadas del ground, marcarlas
            all_ground_orig = self.pipeline.patchwork.getGroundIndices()
            if len(walls) <= len(all_ground_orig):
                wall_global = all_ground_orig[walls] if walls.max() < len(all_ground_orig) else walls
                valid_walls = wall_global[wall_global < N]
                rgb[valid_walls] = COLOR_WALL

        return rgb

    def colorize_stage2(self, points, result):
        """Stage 2: suelo / obstáculo / void / incierto."""
        N = len(points)
        rgb = np.full(N, COLOR_UNCERTAIN, dtype=np.float32)

        rgb[result['ground_mask']] = COLOR_GROUND
        rgb[result['obs_mask']] = COLOR_OBSTACLE
        rgb[result['void_mask']] = COLOR_VOID
        if 'curb_mask' in result and np.any(result['curb_mask']):
            rgb[result['curb_mask']] = COLOR_CURB

        return rgb

    def colorize_clusters(self, points, result):
        """Stage 3: cada cluster DBSCAN con un color diferente. Ground en verde, ruido en gris."""
        N = len(points)
        rgb = np.full(N, COLOR_GROUND, dtype=np.float32)

        cluster_labels = result['cluster_labels']
        obs_mask = result['obs_mask']

        # Puntos obstáculo sin cluster (ruido DBSCAN) → gris
        noise_mask = obs_mask & (cluster_labels == -1)
        rgb[noise_mask] = COLOR_UNCERTAIN

        # Cada cluster con un color de la paleta
        valid_labels = cluster_labels[cluster_labels >= 0]
        if len(valid_labels) > 0:
            unique_clusters = np.unique(valid_labels)
            n_colors = len(CLUSTER_COLORS)
            for cid in unique_clusters:
                mask = cluster_labels == cid
                color = CLUSTER_COLORS[int(cid) % n_colors]
                rgb[mask] = color

        return rgb

    def run_pipeline(self):
        if self.done:
            return
        self.done = True
        self.timer.cancel()

        args = self.args
        seq = args.seq
        stages = args.stages

        self.get_logger().info(f"=== Pipeline Viz: seq {seq}, stages {stages} ===")

        # Inicializar pipeline
        config = PipelineConfig(
            verbose=False,
            enable_delta_r=args.delta_r,
            enable_curb_detection=args.curb,
            curb_height_min=args.curb_min,
            curb_height_max=args.curb_max,
        )
        self.pipeline = LidarPipelineSuite(config)

        # Cargar poses
        info = get_sequence_info(seq)
        poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])

        scan_final = args.scan_end

        # ----------------------------------------------------------------
        # Procesar el frame final a través del pipeline
        # ----------------------------------------------------------------
        points = load_scan(scan_final, seq, curb_labels=args.curb_labels)

        # Pipeline completo: Stage 1+2+3
        result_s3 = self.pipeline.stage3_complete(points)
        result_s2 = self.pipeline.last_stage2_result
        result_s1 = result_s2  # Stage 1 info viene dentro de result_s2

        self.get_logger().info(f"\n  Publicando resultados del frame {scan_final}:")

        if 0 in stages:
            rgb0 = self.colorize_vanilla(points)
            self.pub_vanilla.publish(self.create_rgb_cloud(points, rgb0))
            self.get_logger().info(f"    PW++ vanilla publicado")

        if 1 in stages:
            rgb1 = self.colorize_stage1(points, result_s1)
            self.pub_stage1.publish(self.create_rgb_cloud(points, rgb1))
            n_ground = len(result_s1['ground_indices'])
            n_walls = len(result_s1.get('rejected_walls', []))
            self.get_logger().info(f"    Stage 1: ground={n_ground}, walls={n_walls}")

        if 2 in stages:
            rgb2 = self.colorize_stage2(points, result_s2)
            self.pub_stage2.publish(self.create_rgb_cloud(points, rgb2))
            n_obs = result_s2['obs_mask'].sum()
            n_gnd = result_s2['ground_mask'].sum()
            n_void = result_s2.get('void_mask', np.zeros(0)).sum()
            self.get_logger().info(
                f"    Stage 2: obs={n_obs}, ground={n_gnd}, void={n_void}, "
                f"total={len(points)}"
            )

        if 3 in stages:
            rgb3 = self.colorize_clusters(points, result_s3)
            self.pub_clusters.publish(self.create_rgb_cloud(points, rgb3))
            n_clusters = result_s3.get('n_clusters', 0)
            n_removed = result_s3.get('n_cluster_total_removed', 0)
            self.get_logger().info(
                f"    Stage 3: {n_clusters} clusters, {n_removed} puntos eliminados"
            )

        # Guardar snapshots por stage para republish
        self._snapshots = {
            'vanilla_rgb': rgb0 if 0 in stages else None,
            's1': result_s1,
            's2': result_s2,
            's3': result_s3,
        }

        # Ground truth
        self.publish_gt(scan_final, seq, points)
        # Bordillos GT (cyan=detectado, magenta=perdido)
        self.publish_curb_gt(scan_final, seq, points, result_s3['obs_mask'])

        self.get_logger().info("\n=== Publicado. Mantén RViz abierto. Ctrl+C para salir. ===")
        self.get_logger().info("  Topics: /stage1_cloud /stage2_cloud /gt_cloud")
        self.get_logger().info("  En RViz: Add > By topic > PointCloud2, Color: 'rgb'")

        # Re-publicar periódicamente para que RViz no pierda los mensajes
        self.republish_timer = self.create_timer(2.0, self._republish)
        self._last_points = points
        self._stages = stages

    def _republish(self):
        """Re-publica las nubes periódicamente para que RViz las mantenga."""
        points = self._last_points
        snaps = self._snapshots
        stages = self._stages

        if 0 in stages and snaps.get('vanilla_rgb') is not None:
            self.pub_vanilla.publish(self.create_rgb_cloud(points, snaps['vanilla_rgb']))
        if 1 in stages and snaps.get('s1') is not None:
            rgb1 = self.colorize_stage1(points, snaps['s1'])
            self.pub_stage1.publish(self.create_rgb_cloud(points, rgb1))
        if 2 in stages and snaps.get('s2') is not None:
            rgb2 = self.colorize_stage2(points, snaps['s2'])
            self.pub_stage2.publish(self.create_rgb_cloud(points, rgb2))
        if 3 in stages and snaps.get('s3') is not None:
            rgb3 = self.colorize_clusters(points, snaps['s3'])
            self.pub_clusters.publish(self.create_rgb_cloud(points, rgb3))
        self.publish_gt(self.args.scan_end, self.args.seq, points)
        if snaps.get('s3') is not None:
            self.publish_curb_gt(self.args.scan_end, self.args.seq, points, snaps['s3']['obs_mask'])


def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description='Visualizador del pipeline LiDAR por stages')
    parser.add_argument('--seq', default='04', choices=['00', '04'], help='Secuencia KITTI')
    parser.add_argument('--scan', type=int, default=None, help='Frame único (sobrescribe scan_start/end)')
    parser.add_argument('--scan_start', type=int, default=0, help='Primer frame')
    parser.add_argument('--scan_end', type=int, default=10, help='Último frame')
    parser.add_argument('--stages', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Stages a visualizar (default: 1 2)')
    parser.add_argument('--no-rviz', action='store_true', help='No lanzar RViz automáticamente')
    parser.add_argument('--delta_r', action='store_true', help='Activar delta-r conservador (muestra voids en azul)')
    parser.add_argument('--curb', action='store_true', help='Activar detección de bordillos (cyan en Stage 2)')
    parser.add_argument('--curb_min', type=float, default=0.05, help='Altura mínima bordillo (m)')
    parser.add_argument('--curb_max', type=float, default=0.30, help='Altura máxima bordillo (m)')
    parser.add_argument('--curb_labels', action='store_true',
                        help='Usar datos de 3D-Curb (velodyne + labels de 3d_curb_labels/)')

    clean_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    args = parser.parse_args(clean_args)

    if args.scan is not None:
        args.scan_start = args.scan
        args.scan_end = args.scan

    # Lanzar RViz automáticamente en background
    rviz_proc = None
    if not args.no_rviz:
        rviz_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_viz.rviz')
        rviz_proc = subprocess.Popen(
            ['rviz2', '-d', rviz_config],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[INFO] RViz lanzado (PID {rviz_proc.pid})")

    node = PipelineVizNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if rviz_proc:
            rviz_proc.terminate()


if __name__ == '__main__':
    main()
