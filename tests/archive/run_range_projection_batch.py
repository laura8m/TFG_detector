#!/usr/bin/env python3
"""
Wrapper para ejecutar range_projection.py en modo BATCH (sin ROS spin).

Permite analizar su desempeño (métricas, timing) sin necesidad de RViz.

Uso:
    python3 run_range_projection_batch.py --scan_start 0 --scan_end 4
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Agregar paths necesarios
sys.path.insert(0, '/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
patchwork_path = "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages"
sys.path.insert(0, patchwork_path)

# Importar range_projection como módulo (sin ejecutar main)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "range_projection_module",
    "/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/range_projection.py"
)
range_projection_module = importlib.util.module_from_spec(spec)

# Mock ROS 2 dependencies para evitar errores de import
class MockRclpy:
    def init(self, *args, **kwargs): pass
    def shutdown(self): pass
    def spin(self, node): pass

    class node:
        class Node:
            def __init__(self, name):
                self.name = name
                self.publishers = {}
            def create_publisher(self, msg_type, topic, qos):
                class MockPublisher:
                    def publish(self, msg): pass
                return MockPublisher()
            def create_timer(self, period, callback): pass
            def get_logger(self):
                class MockLogger:
                    def info(self, msg): print(f"[INFO] {msg}")
                    def warn(self, msg): print(f"[WARN] {msg}")
                    def error(self, msg): print(f"[ERROR] {msg}")
                return MockLogger()
            def get_clock(self):
                class MockClock:
                    def now(self):
                        class MockTime:
                            def to_msg(self):
                                class MockStamp:
                                    sec = 0
                                    nanosec = 0
                                return MockStamp()
                        return MockTime()
                return MockClock()
            def destroy_node(self): pass

    class qos:
        class QoSProfile:
            def __init__(self, *args, **kwargs): pass
        class ReliabilityPolicy:
            BEST_EFFORT = 0
            RELIABLE = 1
        class DurabilityPolicy:
            TRANSIENT_LOCAL = 0
            VOLATILE = 1

class MockSensorMsgs:
    class msg:
        class Image:
            def __init__(self):
                self.header = type('obj', (object,), {'stamp': type('obj', (object,), {'sec': 0, 'nanosec': 0})(), 'frame_id': ''})()
        class PointCloud2:
            def __init__(self):
                self.header = type('obj', (object,), {'stamp': type('obj', (object,), {'sec': 0, 'nanosec': 0})(), 'frame_id': ''})()
        class PointField:
            FLOAT32 = 7
            def __init__(self, name='', offset=0, datatype=0, count=0):
                self.name = name
                self.offset = offset
                self.datatype = datatype
                self.count = count

class MockVisualizationMsgs:
    class msg:
        class Marker:
            def __init__(self):
                self.header = type('obj', (object,), {'stamp': type('obj', (object,), {'sec': 0, 'nanosec': 0})(), 'frame_id': ''})()
        class MarkerArray:
            def __init__(self):
                self.markers = []

class MockGeometryMsgs:
    class msg:
        class Point:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
        class TransformStamped:
            def __init__(self):
                self.header = type('obj', (object,), {'stamp': type('obj', (object,), {'sec': 0, 'nanosec': 0})(), 'frame_id': ''})()

class MockStdMsgs:
    class msg:
        class ColorRGBA:
            def __init__(self):
                self.r = 0.0
                self.g = 0.0
                self.b = 0.0
                self.a = 1.0

class MockCvBridge:
    class CvBridge:
        def cv2_to_imgmsg(self, img, encoding): return MockSensorMsgs.msg.Image()

class MockTf2Ros:
    class StaticTransformBroadcaster:
        def __init__(self, node): pass
        def sendTransform(self, transform): pass

# Inyectar mocks en sys.modules ANTES de importar range_projection
sys.modules['rclpy'] = MockRclpy()
sys.modules['rclpy.node'] = MockRclpy.node
sys.modules['rclpy.qos'] = MockRclpy.qos
sys.modules['sensor_msgs'] = MockSensorMsgs()
sys.modules['sensor_msgs.msg'] = MockSensorMsgs.msg
sys.modules['visualization_msgs'] = MockVisualizationMsgs()
sys.modules['visualization_msgs.msg'] = MockVisualizationMsgs.msg
sys.modules['geometry_msgs'] = MockGeometryMsgs()
sys.modules['geometry_msgs.msg'] = MockGeometryMsgs.msg
sys.modules['std_msgs'] = MockStdMsgs()
sys.modules['std_msgs.msg'] = MockStdMsgs.msg
sys.modules['cv_bridge'] = MockCvBridge()
sys.modules['tf2_ros'] = MockTf2Ros()

# Ahora sí, cargar el módulo
try:
    spec.loader.exec_module(range_projection_module)
except Exception as e:
    print(f"Error loading range_projection.py: {e}")
    print("Continuando sin el módulo completo...")


# ================================================================================
# UTILIDADES PARA EVALUACIÓN
# ================================================================================

def load_kitti_scan(scan_id=0):
    """Carga scan KITTI (.bin) y labels SemanticKITTI (.label)."""
    velodyne_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04/velodyne")
    labels_path = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04_labels/04/labels")

    scan_file = velodyne_path / f"{scan_id:06d}.bin"
    points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]

    label_file = labels_path / f"{scan_id:06d}.label"
    labels = np.fromfile(label_file, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF

    return points, semantic_labels


def get_ground_truth_masks(semantic_labels):
    """Genera máscaras de ground truth."""
    obstacle_classes = [
        10, 11, 13, 15, 16, 18, 20,  # Vehicles
        30, 31, 32,  # Persons/cyclists
        50, 51, 52,  # Buildings/fences/walls
        60, 70, 71,  # Trunk, vegetation
        80, 81,  # Poles, traffic signs
        252, 253, 254, 255, 256, 257, 258, 259  # Other movable
    ]
    obstacle_mask = np.isin(semantic_labels, obstacle_classes)
    ground_mask = np.isin(semantic_labels, [40, 44, 48, 49, 72])

    return {
        'obstacle': obstacle_mask,
        'ground': ground_mask
    }


def compute_detection_metrics(gt_mask, pred_mask):
    """Calcula Precision, Recall, F1."""
    TP = np.sum(gt_mask & pred_mask)
    FP = np.sum((~gt_mask) & pred_mask)
    FN = np.sum(gt_mask & (~pred_mask))
    TN = np.sum((~gt_mask) & (~pred_mask))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


# ================================================================================
# MAIN: EJECUTAR RANGE_PROJECTION EN BATCH
# ================================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run range_projection.py in BATCH mode')
    parser.add_argument('--scan_start', type=int, default=0)
    parser.add_argument('--scan_end', type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("RANGE_PROJECTION.PY - BATCH MODE (sin ROS spin)")
    print("=" * 80)
    print(f"Scan range: {args.scan_start} - {args.scan_end}")
    print()

    # Verificar si range_projection_module se cargó correctamente
    if not hasattr(range_projection_module, 'RangeViewNode'):
        print("[ERROR] No se pudo cargar RangeViewNode desde range_projection.py")
        print("[INFO] Intentando análisis directo del algoritmo...")

        # Fallback: implementar algoritmo manualmente basándose en el código leído
        print("[WARN] Implementación fallback NO disponible aún.")
        print("[WARN] Necesitas instalar dependencias ROS 2 correctamente.")
        sys.exit(1)

    # Instanciar el nodo (sin rclpy.init porque usamos mocks)
    try:
        node = range_projection_module.RangeViewNode(
            data_path="/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04",
            scene="04",
            scan_start=args.scan_start,
            scan_end=args.scan_end
        )
        print("[✓] RangeViewNode inicializado correctamente")
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar RangeViewNode: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Procesar frames manualmente (en lugar de usar spin)
    all_metrics = []

    for scan_id in range(args.scan_start, args.scan_end + 1):
        print(f"\n{'='*80}")
        print(f"Procesando scan {scan_id}")
        print(f"{'='*80}")

        # Cargar datos
        points, semantic_labels = load_kitti_scan(scan_id)
        gt_masks = get_ground_truth_masks(semantic_labels)

        print(f"✓ Cargado: {len(points)} puntos, {gt_masks['obstacle'].sum()} GT obstacles")

        # Ejecutar pipeline de range_projection manualmente
        try:
            t_start = time.time()

            # Llamar al método process_scan del nodo (si existe)
            if hasattr(node, 'process_scan'):
                node.current_scan = scan_id
                node.process_scan()
            else:
                print("[WARN] RangeViewNode no tiene método process_scan()")
                print("[INFO] Intentando ejecutar pipeline manualmente...")
                # Aquí habría que llamar a los métodos internos del nodo
                # Pero sin timer_callback es difícil...

            t_end = time.time()
            timing_ms = (t_end - t_start) * 1000.0

            print(f"✓ Pipeline ejecutado en {timing_ms:.1f} ms")

            # Obtener resultados (si están disponibles como atributos del nodo)
            if hasattr(node, 'belief_prob') and node.belief_prob is not None:
                # Belief map → point mask
                # (Necesitarías la lógica inversa de proyección)
                print("[INFO] Belief map disponible, pero falta implementar conversión a point mask")
            else:
                print("[WARN] No se pudo obtener resultados del pipeline")

        except Exception as e:
            print(f"[ERROR] Fallo al procesar scan {scan_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETADO")
    print("=" * 80)


if __name__ == '__main__':
    main()
