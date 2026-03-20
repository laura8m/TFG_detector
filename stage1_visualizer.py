#!/usr/bin/env python3
"""
stage1_visualizer.py
====================
Visualizador RViz para Stage 1 del pipeline LiDAR.

Publica topics ROS 2 para visualizar:
- Ground points (verde)
- Non-ground points (rojo)
- Rejected walls (azul)
- HCD features (colormap)
- Local ground planes (markers)

Uso:
    # Test simple
    python3 stage1_visualizer.py --scan 0

    # Con ablation study (publica 3 configuraciones en topics separados)
    python3 stage1_visualizer.py --scan 0 --ablation

    # Modo continuo (procesa múltiples scans)
    python3 stage1_visualizer.py --scan_range 0 10 --loop

Autor: TFG LiDAR Geometry
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
import struct
from pathlib import Path
import time

# Importar pipeline
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


class Stage1Visualizer(Node):
    """
    Nodo ROS 2 que visualiza resultados del Stage 1 en RViz.
    """

    def __init__(self):
        super().__init__('stage1_visualizer')

        # Publishers con QoS Transient Local para que RViz reciba datos históricos
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Publishers
        self.pub_ground = self.create_publisher(PointCloud2, '/stage1/ground_points', qos)
        self.pub_nonground = self.create_publisher(PointCloud2, '/stage1/nonground_points', qos)
        self.pub_walls = self.create_publisher(PointCloud2, '/stage1/wall_points', qos)
        self.pub_hcd = self.create_publisher(PointCloud2, '/stage1/hcd_colored', qos)
        self.pub_planes = self.create_publisher(MarkerArray, '/stage1/ground_planes', qos)

        # Publishers para ablation study (3 configuraciones)
        self.pub_baseline = self.create_publisher(PointCloud2, '/stage1/ablation/baseline', qos)
        self.pub_wall_rej = self.create_publisher(PointCloud2, '/stage1/ablation/wall_rejection', qos)
        self.pub_hcd_full = self.create_publisher(PointCloud2, '/stage1/ablation/hcd', qos)

        # Storage para mensajes (para republicar)
        self.stored_messages = {}

        # Timer para republicar mensajes cada 0.5 segundos
        self.timer = self.create_timer(0.5, self.republish_callback)

        self.get_logger().info('Stage1Visualizer initialized')

    def republish_callback(self):
        """
        Timer callback que republica mensajes almacenados.
        """
        for topic_name, msg in self.stored_messages.items():
            if 'ground_points' in topic_name:
                self.pub_ground.publish(msg)
            elif 'nonground_points' in topic_name:
                self.pub_nonground.publish(msg)
            elif 'wall_points' in topic_name:
                self.pub_walls.publish(msg)
            elif 'hcd_colored' in topic_name:
                self.pub_hcd.publish(msg)
            elif 'ground_planes' in topic_name:
                self.pub_planes.publish(msg)
            elif 'ablation/baseline' in topic_name:
                self.pub_baseline.publish(msg)
            elif 'ablation/wall_rejection' in topic_name:
                self.pub_wall_rej.publish(msg)
            elif 'ablation/hcd' in topic_name:
                self.pub_hcd_full.publish(msg)

    def publish_pointcloud(self, points, publisher, colors=None, frame_id='velodyne', store_topic=None):
        """
        Publica una nube de puntos en ROS 2.

        Args:
            points: array (N, 3) con coordenadas XYZ
            publisher: Publisher de ROS 2
            colors: array (N, 3) con colores RGB [0-255] (opcional)
            frame_id: frame de referencia TF
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        if colors is not None:
            # PointCloud2 con color RGB
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

            point_step = 16
        else:
            # PointCloud2 sin color (solo XYZ)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            cloud_data = []
            for i in range(len(points)):
                x, y, z = points[i]
                cloud_data.append(struct.pack('fff', x, y, z))

            point_step = 12

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.is_dense = True
        msg.data = b''.join(cloud_data)

        publisher.publish(msg)

        # Almacenar para republicar
        if store_topic:
            self.stored_messages[store_topic] = msg

    def publish_ground_planes(self, local_planes, frame_id='velodyne'):
        """
        Publica markers de los planos ground locales (solo algunos para no saturar).

        Args:
            local_planes: dict {bin_id: {'n': normal, 'd': distance}}
            frame_id: frame de referencia TF
        """
        marker_array = MarkerArray()

        # Seleccionar solo algunos bins para visualizar (cada 10)
        bin_ids = sorted(local_planes.keys())
        selected_bins = bin_ids[::10]  # 1 de cada 10

        for idx, bin_id in enumerate(selected_bins[:50]):  # Max 50 markers
            plane_info = local_planes[bin_id]
            n = plane_info['n']
            d = plane_info['d']

            # Crear marker tipo ARROW (normal del plano)
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'ground_planes'
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Calcular punto central del bin (aproximado)
            # Para simplificar, ponemos la flecha en el origen del plano
            point_on_plane = -d * n

            marker.points.append(Point(x=point_on_plane[0],
                                      y=point_on_plane[1],
                                      z=point_on_plane[2]))
            marker.points.append(Point(x=point_on_plane[0] + n[0] * 0.5,
                                      y=point_on_plane[1] + n[1] * 0.5,
                                      z=point_on_plane[2] + n[2] * 0.5))

            marker.scale.x = 0.05  # Grosor flecha
            marker.scale.y = 0.1   # Grosor punta
            marker.scale.z = 0.0

            # Color según verticalidad (más vertical = más azul)
            nz = abs(n[2])
            marker.color.r = 0.0
            marker.color.g = 1.0 - nz
            marker.color.b = nz
            marker.color.a = 0.7

            marker.lifetime.sec = 0  # Permanente

            marker_array.markers.append(marker)

        self.pub_planes.publish(marker_array)

        # Almacenar para republicar
        self.stored_messages['/stage1/ground_planes'] = marker_array

    def visualize_stage1(self, scan_path, config, publish_suffix=''):
        """
        Ejecuta Stage 1 y publica resultados en RViz.

        Args:
            scan_path: Path al archivo .bin
            config: PipelineConfig
            publish_suffix: Sufijo para topics (para ablation study)
        """
        # Cargar scan
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]

        self.get_logger().info(f'Processing {scan_path.name} ({len(points)} points)...')

        # Ejecutar Stage 1
        pipeline = LidarPipelineSuite(config, data_path=str(scan_path))
        result = pipeline.stage1_complete(points)

        # Extraer resultados
        ground_indices = result['ground_indices']
        nonground_indices = result['nonground_indices']
        rejected_walls = result['rejected_walls']
        hcd = result.get('hcd', None)
        timing_ms = result['timing_ms']

        # Log estadísticas
        self.get_logger().info(
            f"Stage 1 completed in {timing_ms:.1f} ms:\n"
            f"  Ground: {len(ground_indices)} pts\n"
            f"  Non-ground: {len(nonground_indices)} pts\n"
            f"  Walls: {len(rejected_walls)} pts\n"
            f"  HCD: {'computed' if hcd is not None else 'disabled'}"
        )

        # Publicar nubes de puntos coloreadas
        if len(ground_indices) > 0:
            ground_pts = points[ground_indices]
            ground_colors = np.tile([0, 255, 0], (len(ground_pts), 1))  # Verde
            self.publish_pointcloud(ground_pts, self.pub_ground, ground_colors, store_topic='/stage1/ground_points')

        if len(nonground_indices) > 0:
            nonground_pts = points[nonground_indices]
            nonground_colors = np.tile([255, 0, 0], (len(nonground_pts), 1))  # Rojo
            self.publish_pointcloud(nonground_pts, self.pub_nonground, nonground_colors, store_topic='/stage1/nonground_points')

        if len(rejected_walls) > 0:
            wall_pts = points[rejected_walls]
            wall_colors = np.tile([0, 0, 255], (len(wall_pts), 1))  # Azul
            self.publish_pointcloud(wall_pts, self.pub_walls, wall_colors, store_topic='/stage1/wall_points')

        # Publicar HCD coloreado (si está activado)
        if hcd is not None and len(ground_indices) > 0:
            # HCD es un array 1D con valores z_rel normalizados
            z_rel = hcd  # Ya es 1D después de la vectorización
            z_rel_norm = np.clip((z_rel + 1.0) / 2.0, 0, 1)  # Normalizar a [0, 1]

            # Colormap: azul (bajo) -> verde (medio) -> rojo (alto)
            hcd_colors = np.zeros((len(ground_indices), 3), dtype=np.uint8)
            hcd_colors[:, 0] = (z_rel_norm * 255).astype(np.uint8)  # R
            hcd_colors[:, 1] = ((1 - np.abs(z_rel_norm - 0.5) * 2) * 255).astype(np.uint8)  # G
            hcd_colors[:, 2] = ((1 - z_rel_norm) * 255).astype(np.uint8)  # B

            ground_pts = points[ground_indices]
            self.publish_pointcloud(ground_pts, self.pub_hcd, hcd_colors, store_topic='/stage1/hcd_colored')

        # Publicar planos ground (solo algunos)
        if hasattr(pipeline, 'local_planes') and pipeline.local_planes:
            self.publish_ground_planes(pipeline.local_planes)

        return result

    def run_ablation_study_visualized(self, scan_path):
        """
        Ejecuta ablation study y publica cada configuración en un topic separado.

        Args:
            scan_path: Path al archivo .bin
        """
        configs = [
            ('Baseline', PipelineConfig(enable_hybrid_wall_rejection=False,
                                        enable_hcd=False)),
            ('Wall Rejection', PipelineConfig(enable_hybrid_wall_rejection=True,
                                              enable_hcd=False)),
            ('HCD Full', PipelineConfig(enable_hybrid_wall_rejection=True,
                                       enable_hcd=True))
        ]

        publishers = [self.pub_baseline, self.pub_wall_rej, self.pub_hcd_full]

        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]

        self.get_logger().info(f'\n{"="*80}')
        self.get_logger().info(f'ABLATION STUDY: {scan_path.name}')
        self.get_logger().info(f'{"="*80}')

        results = []

        for (name, config), pub in zip(configs, publishers):
            self.get_logger().info(f'\n[Ablation] Testing: {name}')

            pipeline = LidarPipelineSuite(config, data_path=str(scan_path))
            result = pipeline.stage1_complete(points)

            # Publicar nube coloreada
            ground_idx = result['ground_indices']
            nonground_idx = result['nonground_indices']
            wall_idx = result['rejected_walls']

            # Crear nube combinada con colores
            combined_pts = []
            combined_colors = []

            if len(ground_idx) > 0:
                combined_pts.append(points[ground_idx])
                combined_colors.append(np.tile([0, 255, 0], (len(ground_idx), 1)))

            if len(nonground_idx) > 0:
                combined_pts.append(points[nonground_idx])
                combined_colors.append(np.tile([255, 100, 0], (len(nonground_idx), 1)))

            if len(wall_idx) > 0:
                combined_pts.append(points[wall_idx])
                combined_colors.append(np.tile([0, 100, 255], (len(wall_idx), 1)))

            if combined_pts:
                all_pts = np.vstack(combined_pts)
                all_colors = np.vstack(combined_colors)

                # Determinar topic name para storage
                if pub == self.pub_baseline:
                    topic_name = '/stage1/ablation/baseline'
                elif pub == self.pub_wall_rej:
                    topic_name = '/stage1/ablation/wall_rejection'
                else:
                    topic_name = '/stage1/ablation/hcd'

                self.publish_pointcloud(all_pts, pub, all_colors, store_topic=topic_name)

            # Log resultados
            self.get_logger().info(
                f"  Ground: {len(ground_idx)} | "
                f"Non-ground: {len(nonground_idx)} | "
                f"Walls: {len(wall_idx)} | "
                f"Time: {result['timing_ms']:.1f} ms"
            )

            results.append({
                'config': name,
                'ground': len(ground_idx),
                'nonground': len(nonground_idx),
                'walls': len(wall_idx),
                'timing_ms': result['timing_ms']
            })

        # Tabla comparativa
        self.get_logger().info(f'\n{"="*80}')
        self.get_logger().info('ABLATION STUDY RESULTS')
        self.get_logger().info(f'{"="*80}')
        for r in results:
            self.get_logger().info(
                f"{r['config']:20s} | "
                f"Ground: {r['ground']:6d} | "
                f"Non-ground: {r['nonground']:6d} | "
                f"Walls: {r['walls']:5d} | "
                f"Time: {r['timing_ms']:6.1f} ms"
            )
        self.get_logger().info(f'{"="*80}\n')

        return results


def main(args=None):
    """
    Main CLI para visualizar Stage 1 en RViz.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Stage 1 Visualizer - RViz Publisher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Visualizar scan 0 con configuración completa
  python3 stage1_visualizer.py --scan 0

  # Ablation study (publica 3 configs en topics separados)
  python3 stage1_visualizer.py --scan 0 --ablation

  # Procesar scans en loop
  python3 stage1_visualizer.py --scan_range 0 4 --loop --delay 2.0

Topics publicados:
  /stage1/ground_points        - Puntos ground (verde)
  /stage1/nonground_points     - Puntos non-ground (rojo)
  /stage1/wall_points          - Paredes rechazadas (azul)
  /stage1/hcd_colored          - HCD coloreado por z_rel
  /stage1/ground_planes        - Normales de planos locales

Topics ablation:
  /stage1/ablation/baseline         - Baseline (solo Patchwork++)
  /stage1/ablation/wall_rejection   - + Hybrid Wall Rejection
  /stage1/ablation/hcd              - + HCD completo
        """
    )

    parser.add_argument('--scan', type=int, default=0,
                        help='Scan ID to visualize')
    parser.add_argument('--scan_range', type=int, nargs=2, default=None,
                        help='Range of scans (start, end) for loop mode')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study (publishes 3 configs)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop through scans continuously')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between scans in loop mode (seconds)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Custom data path (default: data_kitti/04/04/velodyne/)')
    parser.add_argument('--enable_hcd', action='store_true',
                        help='Enable HCD in single test mode')
    parser.add_argument('--disable_wall_rejection', action='store_true',
                        help='Disable wall rejection in single test mode')

    cli_args = parser.parse_args()

    # Inicializar ROS 2
    rclpy.init(args=args)
    node = Stage1Visualizer()

    try:
        # Determinar rango de scans
        if cli_args.scan_range:
            scan_ids = range(cli_args.scan_range[0], cli_args.scan_range[1] + 1)
        else:
            scan_ids = [cli_args.scan]

        # Path base
        if cli_args.data_path:
            base_path = Path(cli_args.data_path)
        else:
            from data_paths import get_velodyne_dir
            base_path = get_velodyne_dir('04')

        # Modo ablation
        if cli_args.ablation:
            for scan_id in scan_ids:
                scan_path = base_path / f'{scan_id:06d}.bin'
                if not scan_path.exists():
                    node.get_logger().warn(f'Scan {scan_id:06d} not found, skipping')
                    continue

                node.run_ablation_study_visualized(scan_path)

                if cli_args.loop and len(scan_ids) > 1:
                    time.sleep(cli_args.delay)
                elif not cli_args.loop:
                    break

        # Modo simple
        else:
            config = PipelineConfig(
                enable_hcd=cli_args.enable_hcd,
                enable_hybrid_wall_rejection=not cli_args.disable_wall_rejection,
                verbose=True
            )

            for scan_id in scan_ids:
                scan_path = base_path / f'{scan_id:06d}.bin'
                if not scan_path.exists():
                    node.get_logger().warn(f'Scan {scan_id:06d} not found, skipping')
                    continue

                node.visualize_stage1(scan_path, config)

                if cli_args.loop and len(scan_ids) > 1:
                    time.sleep(cli_args.delay)
                elif not cli_args.loop:
                    break

        # Mantener nodo vivo para que RViz pueda recibir los mensajes
        node.get_logger().info('\nPublished to RViz. Press Ctrl+C to exit.')
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
