#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import os
import argparse
from pathlib import Path
import time

# Importar nuestra suite modular
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lidar_modules import LidarProcessingSuite

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class LidarNode(Node):
    def __init__(self, data_root, scene, scan_start, n_scans):
        super().__init__('lidar_sota_node')
        
        # Broadcaster para evitar errores de TF en RViz
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()
        
        self.data_root = Path(data_root)
        self.scene = scene
        self.scan_start = scan_start
        self.n_scans = n_scans
        
        # Verificar setup
        self.get_logger().info(f"Iniciando Nodo SOTA LiDAR...")
        self.get_logger().info(f"Scene: {scene}, Start: {scan_start}, N Scans: {n_scans}")
        
        # Inicializar Pipeline (con path dummy inicial, se actualizará)
        # Usamos el primer scan para init
        first_scan_path = self.get_scan_path(scan_start)
        if not first_scan_path:
            self.get_logger().error("No se pudo encontrar el primer scan.")
            sys.exit(1)
            
        self.pipeline = LidarProcessingSuite(str(first_scan_path), sensor_height=1.73, ros_node=self)
        
        # --- BATCH PROCESSING ---
        # Ejecutar el filtro de Bayes para N scans
        self.run_batch()
        
        # Timer para republicar el resultado final (1 Hz)
        self.timer = self.create_timer(1.0, self.republish)
        
    def broadcast_static_tf(self):
        """Publica una transformada identidad estática de map -> velodyne."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'velodyne'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)
        
    def get_scan_path(self, scan_idx):
        """Construye la ruta al archivo .bin para un índice dado."""
        scan_str = f"{int(scan_idx):06d}"
        # Intentar estructura SemanticKITTI standard
        # dataset/sequences/XX/velodyne/YYYYYY.bin
        p = self.data_root / "sequences" / self.scene / "velodyne" / f"{scan_str}.bin"
        if p.exists(): return p
        
        # Intentar estructura plana (si data_root apunta directo a velodyne?)
        p = self.data_root / f"{scan_str}.bin"
        if p.exists(): return p
        
        return None

    def run_batch(self):
        """Ejecuta el pipeline secuencialmente para N scans."""
        self.get_logger().info("--- Iniciando Batch Processing ---")
        t0 = time.time()
        
        for i in range(self.n_scans):
            scan_idx = self.scan_start + i
            path = self.get_scan_path(scan_idx)
            
            if not path:
                self.get_logger().warn(f"Scan {scan_idx} no encontrado. Abortando batch.")
                break
                
            self.get_logger().info(f"Procesando Scan {scan_idx} ({i+1}/{self.n_scans})...")
            
            # Actualizar path y scan index en pipeline
            self.pipeline.data_path = str(path)
            self.pipeline.current_scan = scan_idx
            
            # Ejecutar update
            self.pipeline.run_full_pipeline()
            
        elapsed = time.time() - t0
        self.get_logger().info(f"--- Batch Completado en {elapsed:.2f}s ---")

    def republish(self):
        """Republica la última visualización para mantener RViz vivo."""
        # self.get_logger().info("Republicando visualización...")
        if hasattr(self.pipeline, 'republish_last'):
            self.pipeline.republish_last()

def main(args=None):
    rclpy.init(args=args)
    
    # Argument Parsing like range_projection.py
    parser = argparse.ArgumentParser(description='Run Lidar Node with Bayes Filter Batch')
    parser.add_argument('--data_path', default='/home/insia/lidar_ws/data_odometry_velodyne/dataset', 
                        help='Path to SemanticKITTI dataset root')
    parser.add_argument('--scene', default='00', help='Sequence Number')
    parser.add_argument('--scan', default='0', help='Start scan index')
    parser.add_argument('--n_scans', default='5', help='Number of scans to process (Batch size)')
    
    # Filter ROS args
    clean_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    parsed_args = parser.parse_args(clean_args)
    
    node = LidarNode(
        data_root=parsed_args.data_path,
        scene=parsed_args.scene,
        scan_start=int(parsed_args.scan),
        n_scans=int(parsed_args.n_scans)
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
