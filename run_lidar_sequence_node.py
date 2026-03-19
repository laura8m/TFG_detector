#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Importar nuestra suite modular
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lidar_modules import LidarProcessingSuite

class LidarSequenceNode(Node):
    def __init__(self, data_path, scene, scan_start, scan_end):
        super().__init__('lidar_sota_seq_node')
        
        self.data_root = Path(data_path)
        self.scene = scene
        self.current_scan = scan_start
        self.scan_end = scan_end
        
        self.get_logger().info(f"=== Iniciando Secuencia SemanticKITTI ===")
        self.get_logger().info(f"Scene: {self.scene}")
        self.get_logger().info(f"Range: {scan_start} -> {scan_end}")
        
        # 1. Construir ruta inicial
        first_file = self.get_file_path(self.current_scan)
        if not first_file.exists():
            self.get_logger().error(f"No se encontró el archivo inicial: {first_file}")
            sys.exit(1)
            
        # 2. Inicializar Pipeline (Persistente para filtro Bayesiano)
        # Se pasa 'self' como ros_node para que publique automáticamente
        self.pipeline = LidarProcessingSuite(str(first_file), sensor_height=1.73, ros_node=self)
        
        # 3. Timer para reproducción (e.g., 2Hz para ver algo, aunque el filtro funciona mejor a 10Hz)
        self.timer = self.create_timer(0.2, self.process_next_frame) # 5 Hz

    def get_file_path(self, scan_idx):
        """Construye la ruta al archivo .bin de SemanticKITTI."""
        # Estructura: dataset/sequences/00/velodyne/000000.bin
        return self.data_root / "sequences" / self.scene / "velodyne" / f"{scan_idx:06d}.bin"

    def process_next_frame(self):
        if self.current_scan > self.scan_end:
            self.get_logger().info("--- Secuencia Finalizada ---")
            # Reiniciar para loop visual o detener? 
            # Detenemos para que el usuario inspeccione el resultado final del filtro.
            self.timer.cancel()
            return

        file_path = self.get_file_path(self.current_scan)
        if not file_path.exists():
            self.get_logger().warn(f"Archivo no encontrado: {file_path}. Saltando...")
            self.current_scan += 1
            return

        self.get_logger().info(f"Procesando Frame {self.current_scan:06d}...")
        
        # Actualizar ruta en el pipeline
        self.pipeline.data_path = str(file_path)
        
        # EJECUTAR PIPELINE (Actualiza filtro Bayesiano y Publica)
        self.pipeline.run_full_pipeline()
        
        self.current_scan += 1

def main(args=None):
    rclpy.init(args=args)
    
    # Parsear Argumentos estilo range_projection.py
    parser = argparse.ArgumentParser(description='Ejecutar Pipeline SOTA en Secuencia SemanticKITTI')
    parser.add_argument('--data_path', default='/home/insia/lidar_ws/data_odometry_velodyne/dataset', 
                        help='Ruta raíz del dataset (contiene sequences/)')
    parser.add_argument('--scene', default='00', help='Número de secuencia (ej. 00)')
    parser.add_argument('--scan_start', default=0, type=int, help='Índice de inicio')
    parser.add_argument('--scan_end', default=4, type=int, help='Índice final')
    
    # Filtrar argumentos de ROS
    clean_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    parsed_args = parser.parse_args(clean_args)

    node = LidarSequenceNode(
        parsed_args.data_path,
        parsed_args.scene,
        parsed_args.scan_start,
        parsed_args.scan_end
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
