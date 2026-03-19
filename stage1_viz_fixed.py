#!/usr/bin/env python3
"""
Stage 1 Visualizer - Versión simplificada y funcional
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from pathlib import Path

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


class Stage1VisualizerSimple(Node):
    def __init__(self):
        super().__init__('stage1_viz_simple')

        # Publishers simples (sin QoS especial)
        self.pub_ground = self.create_publisher(PointCloud2, '/stage1/ground_points', 10)
        self.pub_nonground = self.create_publisher(PointCloud2, '/stage1/nonground_points', 10)
        self.pub_walls = self.create_publisher(PointCloud2, '/stage1/wall_points', 10)

        # Timer para republicar cada 0.5 segundos
        self.timer = self.create_timer(0.5, self.republish)

        # Storage
        self.msg_ground = None
        self.msg_nonground = None
        self.msg_walls = None

        self.get_logger().info('Stage1VisualizerSimple initialized')

    def republish(self):
        """Republica mensajes almacenados"""
        if self.msg_ground is not None:
            self.pub_ground.publish(self.msg_ground)
        if self.msg_nonground is not None:
            self.pub_nonground.publish(self.msg_nonground)
        if self.msg_walls is not None:
            self.pub_walls.publish(self.msg_walls)

    def create_pointcloud_msg(self, points, colors):
        """Crea mensaje PointCloud2 con colores RGB"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            cloud_data.append(struct.pack('fffI', x, y, z, rgb))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(points)
        msg.is_dense = True
        msg.data = b''.join(cloud_data)

        return msg

    def process_scan(self, scan_path, config):
        """Procesa un scan y publica resultados"""
        self.get_logger().info(f'Processing {scan_path}...')

        # Cargar scan
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]

        self.get_logger().info(f'Loaded {len(points)} points')

        # Ejecutar Stage 1
        pipeline = LidarPipelineSuite(config, data_path=str(scan_path))
        result = pipeline.stage1_complete(points)

        # Extraer resultados
        ground_idx = result['ground_indices']
        nonground_idx = result['nonground_indices']
        walls_idx = result['rejected_walls']

        self.get_logger().info(
            f"Stage 1 done: Ground={len(ground_idx)}, "
            f"NonGround={len(nonground_idx)}, Walls={len(walls_idx)}"
        )

        # Crear mensajes
        if len(ground_idx) > 0:
            ground_pts = points[ground_idx]
            ground_colors = np.tile([0, 255, 0], (len(ground_pts), 1))  # Verde
            self.msg_ground = self.create_pointcloud_msg(ground_pts, ground_colors)
            self.get_logger().info(f'Created ground cloud: {len(ground_pts)} points')

        if len(nonground_idx) > 0:
            nonground_pts = points[nonground_idx]
            nonground_colors = np.tile([255, 0, 0], (len(nonground_pts), 1))  # Rojo
            self.msg_nonground = self.create_pointcloud_msg(nonground_pts, nonground_colors)
            self.get_logger().info(f'Created nonground cloud: {len(nonground_pts)} points')

        if len(walls_idx) > 0:
            walls_pts = points[walls_idx]
            walls_colors = np.tile([0, 0, 255], (len(walls_pts), 1))  # Azul
            self.msg_walls = self.create_pointcloud_msg(walls_pts, walls_colors)
            self.get_logger().info(f'Created walls cloud: {len(walls_pts)} points')

        # Publicar inmediatamente
        self.republish()
        self.get_logger().info('Messages published. Ctrl+C to exit.')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1 Visualizer - Fixed')
    parser.add_argument('--scan', type=int, default=0, help='Scan ID')
    parser.add_argument('--sequence', type=str, default='00', help='KITTI sequence (00, 04, etc.)')
    parser.add_argument('--enable_hcd', action='store_true', help='Enable HCD')
    parser.add_argument('--disable_wall_rejection', action='store_true', help='Disable wall rejection')
    args = parser.parse_args()

    # Config
    config = PipelineConfig(
        enable_hcd=args.enable_hcd,
        enable_hybrid_wall_rejection=not args.disable_wall_rejection,
        verbose=True
    )

    # Path
    scan_path = Path(f'/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/{args.sequence}/{args.sequence}/velodyne/{args.scan:06d}.bin')

    if not scan_path.exists():
        print(f'ERROR: {scan_path} not found')
        return

    # Init ROS
    rclpy.init()
    node = Stage1VisualizerSimple()

    # Process scan
    node.process_scan(scan_path, config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
