#!/usr/bin/env python3
"""
Test simple para verificar que RViz recibe mensajes.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_test_publisher')

        # Publisher simple
        self.pub = self.create_publisher(PointCloud2, '/test_cloud', 10)

        # Timer para publicar cada 1 segundo
        self.timer = self.create_timer(1.0, self.publish_test_cloud)

        self.get_logger().info('Simple publisher initialized')

    def publish_test_cloud(self):
        """Publica una nube de puntos de prueba (cubo simple)"""

        # Crear puntos de un cubo 10x10x10
        points = []
        for x in range(-5, 6, 2):
            for y in range(-5, 6, 2):
                for z in range(0, 6, 2):
                    points.append([float(x), float(y), float(z)])

        points = np.array(points, dtype=np.float32)

        # Crear mensaje PointCloud2
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = []
        for pt in points:
            cloud_data.append(struct.pack('fff', pt[0], pt[1], pt[2]))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = b''.join(cloud_data)

        self.pub.publish(msg)
        self.get_logger().info(f'Published {len(points)} test points')

def main():
    rclpy.init()
    node = SimplePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
