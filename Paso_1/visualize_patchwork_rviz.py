#!/usr/bin/env python3
"""
Nodo ROS 2 para visualizar resultados de Patchwork++ vanilla en RViz.
Muestra claramente las paredes clasificadas incorrectamente como suelo.

Uso:
    Terminal 1: ros2 run python3 visualize_patchwork_rviz.py
    Terminal 2: rviz2 -d ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea/patchwork_debug.rviz
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import sys
from pathlib import Path
import struct

# Añadir Patchwork++ al path

class PatchworkVisualizer(Node):
    def __init__(self):
        super().__init__('patchwork_visualizer')

        # Publishers
        self.pub_raw = self.create_publisher(PointCloud2, '/patchwork_raw', 10)
        self.pub_ground = self.create_publisher(PointCloud2, '/patchwork_ground', 10)
        self.pub_nonground = self.create_publisher(PointCloud2, '/patchwork_nonground', 10)
        self.pub_suspicious = self.create_publisher(PointCloud2, '/patchwork_suspicious_walls', 10)
        self.pub_vertical_segments = self.create_publisher(PointCloud2, '/patchwork_vertical_segments', 10)

        # load_and_process() se llama desde main() después de configurar scan_id

        # Timer para republicar (1 Hz)
        self.timer = self.create_timer(1.0, self.publish_clouds)

        self.get_logger().info("="*80)
        self.get_logger().info("Patchwork++ Visualizer iniciado")
        self.get_logger().info("="*80)
        self.get_logger().info("")
        self.get_logger().info("Abre RViz con:")
        self.get_logger().info("  rviz2 -d ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea/patchwork_debug.rviz")
        self.get_logger().info("")
        self.get_logger().info("Topics publicados:")
        self.get_logger().info("  /patchwork_raw                 - Nube original (blanco)")
        self.get_logger().info("  /patchwork_ground              - Puntos 'suelo' (verde)")
        self.get_logger().info("  /patchwork_nonground           - Puntos 'no-suelo' (rojo)")
        self.get_logger().info("  /patchwork_suspicious_walls    - Puntos elevados Z>0 (azul)")
        self.get_logger().info("  /patchwork_vertical_segments   - Segmentos verticales ΔZ>0.5m (magenta)")
        self.get_logger().info("")
        self.get_logger().info("="*80)

    def load_and_process(self):
        """Cargar datos y ejecutar Patchwork++"""
        import pypatchworkpp

        # Cargar datos (scan configurable via argumento --scan, default 0)
        scan_id = getattr(self, 'scan_id', 0)
        bin_file = Path(f"/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/00/00/velodyne/{scan_id:06d}.bin")
        scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
        self.points_raw = scan[:, :3]

        self.get_logger().info(f"Cargados {len(self.points_raw):,} puntos desde {bin_file.name}")

        # Ejecutar Patchwork++ (configuración vanilla)
        params = pypatchworkpp.Parameters()
        params.verbose = False
        patchwork = pypatchworkpp.patchworkpp(params)
        patchwork.estimateGround(self.points_raw)

        self.points_ground = patchwork.getGround()
        self.points_nonground = patchwork.getNonground()

        self.get_logger().info(f"Clasificación: {len(self.points_ground):,} suelo, {len(self.points_nonground):,} no-suelo")

        # Detectar puntos sospechosos (Z > 0)
        self.points_suspicious = self.points_ground[self.points_ground[:, 2] > 0.0]

        self.get_logger().info(f"⚠️  Puntos sospechosos (Z > 0): {len(self.points_suspicious):,}")

        # Detectar segmentos verticales
        self.detect_vertical_segments()

    def detect_vertical_segments(self):
        """Detectar segmentos con alta variación vertical (paredes)"""
        r = np.sqrt(self.points_ground[:, 0]**2 + self.points_ground[:, 1]**2)
        theta = np.arctan2(self.points_ground[:, 1], self.points_ground[:, 0])
        z = self.points_ground[:, 2]

        # Bins cilíndricos
        r_bins = np.arange(0, 80, 1.0)
        theta_bins = np.arange(-np.pi, np.pi, np.radians(5))

        r_idx = np.digitize(r, r_bins)
        theta_idx = np.digitize(theta, theta_bins)

        # Analizar cada bin
        vertical_points = []

        for r_id in range(len(r_bins)):
            for theta_id in range(len(theta_bins)):
                mask = (r_idx == r_id) & (theta_idx == theta_id)
                bin_points = self.points_ground[mask]

                if len(bin_points) < 5:
                    continue

                z_min = np.min(bin_points[:, 2])
                z_max = np.max(bin_points[:, 2])
                delta_z = z_max - z_min

                # Si ΔZ > 0.5m → segmento vertical (pared)
                if delta_z > 0.5:
                    vertical_points.append(bin_points)

        if len(vertical_points) > 0:
            self.points_vertical = np.vstack(vertical_points)
            self.get_logger().info(f"⚠️  Segmentos verticales detectados: {len(self.points_vertical):,} puntos en {len(vertical_points)} segmentos")
        else:
            self.points_vertical = np.zeros((0, 3), dtype=np.float32)

    def create_cloud_rgb(self, points, r, g, b):
        """Crear PointCloud2 con color RGB (compatible RViz)"""
        if len(points) == 0:
            # Nube vacía
            msg = PointCloud2()
            msg.header = Header()
            msg.header.frame_id = 'velodyne'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height = 1
            msg.width = 0
            return msg

        msg = PointCloud2()
        msg.header = Header()
        msg.header.frame_id = 'velodyne'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.height = 1
        msg.width = len(points)

        # Usar formato UINT32 para RGB (más compatible)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True

        # Empaquetar RGB como uint32 (R=bits 16-23, G=bits 8-15, B=bits 0-7)
        rgb_uint32 = ((r.astype(np.uint32) << 16) |
                      (g.astype(np.uint32) << 8) |
                      b.astype(np.uint32))

        # Crear array estructurado
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        cloud_data['rgb'] = rgb_uint32

        msg.data = cloud_data.tobytes()

        return msg

    def publish_clouds(self):
        """Publicar todas las nubes de puntos"""
        stamp = self.get_clock().now().to_msg()

        # 1. Nube original (blanco)
        msg_raw = self.create_cloud_rgb(
            self.points_raw,
            np.full(len(self.points_raw), 200, dtype=np.uint8),
            np.full(len(self.points_raw), 200, dtype=np.uint8),
            np.full(len(self.points_raw), 200, dtype=np.uint8)
        )
        msg_raw.header.stamp = stamp
        self.pub_raw.publish(msg_raw)

        # 2. Suelo (verde)
        msg_ground = self.create_cloud_rgb(
            self.points_ground,
            np.zeros(len(self.points_ground), dtype=np.uint8),
            np.full(len(self.points_ground), 255, dtype=np.uint8),
            np.zeros(len(self.points_ground), dtype=np.uint8)
        )
        msg_ground.header.stamp = stamp
        self.pub_ground.publish(msg_ground)

        # 3. No-suelo (rojo)
        msg_nonground = self.create_cloud_rgb(
            self.points_nonground,
            np.full(len(self.points_nonground), 255, dtype=np.uint8),
            np.zeros(len(self.points_nonground), dtype=np.uint8),
            np.zeros(len(self.points_nonground), dtype=np.uint8)
        )
        msg_nonground.header.stamp = stamp
        self.pub_nonground.publish(msg_nonground)

        # 4. Puntos sospechosos Z > 0 (azul brillante)
        if len(self.points_suspicious) > 0:
            msg_suspicious = self.create_cloud_rgb(
                self.points_suspicious,
                np.zeros(len(self.points_suspicious), dtype=np.uint8),
                np.full(len(self.points_suspicious), 100, dtype=np.uint8),
                np.full(len(self.points_suspicious), 255, dtype=np.uint8)
            )
            msg_suspicious.header.stamp = stamp
            self.pub_suspicious.publish(msg_suspicious)

        # 5. Segmentos verticales (magenta)
        if len(self.points_vertical) > 0:
            msg_vertical = self.create_cloud_rgb(
                self.points_vertical,
                np.full(len(self.points_vertical), 255, dtype=np.uint8),
                np.zeros(len(self.points_vertical), dtype=np.uint8),
                np.full(len(self.points_vertical), 255, dtype=np.uint8)
            )
            msg_vertical.header.stamp = stamp
            self.pub_vertical_segments.publish(msg_vertical)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=int, default=0, help='Scan number')
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = PatchworkVisualizer()
    node.scan_id = parsed.scan
    node.load_and_process()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
