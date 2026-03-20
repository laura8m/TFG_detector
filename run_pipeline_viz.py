#!/usr/bin/env python3
"""
Visualizador del pipeline LiDAR por stages en RViz.

Publica la nube de puntos coloreada por clasificación después de cada stage,
para ver la evolución del pipeline.

Pipeline de 3 stages:
  1. Ground estimation (Patchwork++ + wall rejection)
  2. Delta-r anomaly detection
  3. DBSCAN clustering + hull generation

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

def load_scan(scan_id, seq):
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
# Colores por categoría SemanticKITTI para Ground Truth
GT_LABEL_COLORS = {
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
        self.pub_stage1 = self.create_publisher(PointCloud2, '/stage1_cloud', 10)
        self.pub_stage2 = self.create_publisher(PointCloud2, '/stage2_cloud', 10)
        self.pub_stage3 = self.create_publisher(PointCloud2, '/stage3_cloud', 10)
        self.pub_gt     = self.create_publisher(PointCloud2, '/gt_cloud', 10)

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
        """Publica ground truth de SemanticKITTI coloreado por etiqueta."""
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

        return rgb

    def colorize_stage3(self, points, result):
        """Stage 3: clusters coloreados (DBSCAN)."""
        N = len(points)
        rgb = np.full(N, COLOR_UNCERTAIN, dtype=np.float32)

        ground_idx = result.get('ground_indices', np.array([]))
        if len(ground_idx) > 0:
            rgb[ground_idx] = COLOR_GROUND

        obs = result['obs_mask']
        if 'cluster_labels' in result and np.any(obs):
            labels = result['cluster_labels']
            unique_labels = set(labels[obs]) - {-1}

            # Generar colores distintos por cluster
            import colorsys
            n_colors = max(len(unique_labels), 1)
            cluster_colors = {}
            for i, label in enumerate(sorted(unique_labels)):
                h = i / n_colors
                r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
                cluster_colors[label] = rgb_to_float(int(r*255), int(g*255), int(b*255))

            for idx in np.where(obs)[0]:
                lbl = labels[idx]
                if lbl in cluster_colors:
                    rgb[idx] = cluster_colors[lbl]
                else:
                    rgb[idx] = COLOR_OBSTACLE
        else:
            rgb[obs] = COLOR_OBSTACLE

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
        config = PipelineConfig(verbose=False)
        self.pipeline = LidarPipelineSuite(config)

        # Cargar poses
        info = get_sequence_info(seq)
        poses = LidarPipelineSuite.load_kitti_poses(info['poses_file'])

        scan_final = args.scan_end

        # ----------------------------------------------------------------
        # Procesar el frame final a través del pipeline
        # ----------------------------------------------------------------
        points = load_scan(scan_final, seq)

        # Stage 1+2+3 completo (stage3_complete ejecuta stage1+2+3 internamente)
        if 3 in stages:
            result_s3 = self.pipeline.stage3_complete(points)

            # Snapshot Stage 2: usar el resultado que stage3 calculó internamente
            result_s2 = self.pipeline.last_stage2_result
        else:
            # Solo stages 1 y/o 2 — no necesitamos Stage 3
            result_s2 = self.pipeline.stage2_complete(points)
            result_s3 = None

        # Stage 1 info viene dentro de result_s2
        result_s1 = result_s2

        self.get_logger().info(f"\n  Publicando resultados del frame {scan_final}:")

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
            n_unc = result_s2.get('uncertain_mask', np.zeros(0)).sum()
            self.get_logger().info(
                f"    Stage 2: obs={n_obs}, ground={n_gnd}, void={n_void}, "
                f"uncertain={n_unc}, total={len(points)}"
            )

        if 3 in stages and result_s3 is not None:
            rgb3 = self.colorize_stage3(points, result_s3)
            self.pub_stage3.publish(self.create_rgb_cloud(points, rgb3))
            self.get_logger().info(
                f"    Stage 3: clusters={result_s3.get('n_clusters', 0)}, "
                f"removed={result_s3.get('n_cluster_total_removed', 0)}"
            )

        # Guardar snapshots por stage para republish
        self._snapshots = {
            's1': result_s1,
            's2': result_s2,
            's3': result_s3,
        }

        # Ground truth
        self.publish_gt(scan_final, seq, points)

        self.get_logger().info("\n=== Publicado. Mantén RViz abierto. Ctrl+C para salir. ===")
        self.get_logger().info("  Topics: /stage1_cloud /stage2_cloud /stage3_cloud /gt_cloud")
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

        if 1 in stages and snaps.get('s1') is not None:
            rgb1 = self.colorize_stage1(points, snaps['s1'])
            self.pub_stage1.publish(self.create_rgb_cloud(points, rgb1))
        if 2 in stages and snaps.get('s2') is not None:
            rgb2 = self.colorize_stage2(points, snaps['s2'])
            self.pub_stage2.publish(self.create_rgb_cloud(points, rgb2))
        if 3 in stages and snaps.get('s3') is not None:
            rgb3 = self.colorize_stage3(points, snaps['s3'])
            self.pub_stage3.publish(self.create_rgb_cloud(points, rgb3))

        self.publish_gt(self.args.scan_end, self.args.seq, points)


def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description='Visualizador del pipeline LiDAR por stages')
    parser.add_argument('--seq', default='04', choices=['00', '04'], help='Secuencia KITTI')
    parser.add_argument('--scan', type=int, default=None, help='Frame único (sobrescribe scan_start/end)')
    parser.add_argument('--scan_start', type=int, default=0, help='Primer frame')
    parser.add_argument('--scan_end', type=int, default=10, help='Último frame')
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3],
                        help='Stages a visualizar (default: 1 2 3)')
    parser.add_argument('--no-rviz', action='store_true', help='No lanzar RViz automáticamente')

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
