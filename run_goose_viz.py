#!/usr/bin/env python3
"""
Visualización de un frame GOOSE en RViz con detección de bordillos.

Publica:
  /goose_stage1   — Stage 1 (PW++ + WR): verde=ground, rojo=obstáculo
  /goose_curb     — Stage 1 + curb: verde=ground, rojo=obstáculo, cyan=bordillo
  /goose_curb_gt  — Ground truth de bordillos: cyan=detectado, magenta=perdido

Uso:
  python3 run_goose_viz.py
  python3 run_goose_viz.py --curb_min 0.05 --curb_max 0.30
"""

import numpy as np
import struct
import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig


# ========================================
# GOOSE labels
# ========================================
GOOSE_CURB = 22

GOOSE_OBSTACLE = np.array([
    1, 4, 6, 10, 12, 13, 14, 15, 19, 20, 22,
    25, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 45, 46, 47, 48, 49, 57, 58, 60, 61,
    16, 17, 27, 51, 52, 59, 62,
], dtype=np.uint32)

GOOSE_IGNORE = np.array([0, 2, 5, 8, 18, 29, 30, 43, 44, 53, 54, 55, 56], dtype=np.uint32)


# ========================================
# Colores
# ========================================
def rgb_to_float(r, g, b):
    rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', rgb_int))[0]

COLOR_GROUND   = rgb_to_float(50, 200, 50)
COLOR_OBSTACLE = rgb_to_float(230, 50, 50)
COLOR_CURB     = rgb_to_float(0, 230, 230)
COLOR_CURB_MISS = rgb_to_float(230, 0, 230)


class GooseViz(Node):
    def __init__(self, args):
        super().__init__('goose_viz')
        self.args = args

        self.pub_stage1 = self.create_publisher(PointCloud2, '/goose_stage1', 10)
        self.pub_curb = self.create_publisher(PointCloud2, '/goose_curb', 10)
        self.pub_curb_gt = self.create_publisher(PointCloud2, '/goose_curb_gt', 10)

        self.timer = self.create_timer(1.0, self.run)
        self.done = False

    def create_rgb_cloud(self, points, rgb_values):
        N = len(points)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        data = np.zeros(N, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['rgb'] = rgb_values

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = N
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * N
        msg.data = data.tobytes()
        msg.is_dense = True
        return msg

    def run(self):
        if self.done:
            # Republica cada segundo para que RViz lo pille
            if hasattr(self, '_msg1'):
                self.pub_stage1.publish(self._msg1)
                self.pub_curb.publish(self._msg2)
                self.pub_curb_gt.publish(self._msg3)
            return
        self.done = True

        args = self.args
        sample_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'goose_sample'
        bin_files = sorted((sample_dir / 'velodyne').glob('*.bin'))
        label_files = sorted((sample_dir / 'labels').glob('*.label'))

        if not bin_files or not label_files:
            self.get_logger().error('No se encontraron archivos en goose_sample/')
            return

        bin_file = bin_files[0]
        label_file = label_files[0]

        self.get_logger().info(f'Cargando: {bin_file.name}')

        pts = np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl = np.fromfile(str(label_file), dtype=np.uint32) & 0xFFFF
        N = min(len(pts), len(lbl))
        pts, lbl = pts[:N], lbl[:N]

        gt_curb = lbl == GOOSE_CURB
        gt_obs = np.isin(lbl, GOOSE_OBSTACLE)
        valid = ~np.isin(lbl, GOOSE_IGNORE)

        self.get_logger().info(f'  {N} puntos, {gt_curb.sum()} GT curb')

        # --- Stage 1: sin curb ---
        cfg_no = PipelineConfig(
            sensor_height=2.089, n_rings=128, fov_up_deg=15.0, fov_down_deg=-15.0,
            enable_hybrid_wall_rejection=True, enable_curb_detection=False,
            enable_delta_r=False, verbose=False)
        pipe_no = LidarPipelineSuite(cfg_no)
        r1 = pipe_no.stage2_complete(pts)

        rgb1 = np.full(N, COLOR_GROUND, dtype=np.float32)
        rgb1[r1['obs_mask']] = COLOR_OBSTACLE

        # --- Stage 1 + curb ---
        cfg_curb = PipelineConfig(
            sensor_height=2.089, n_rings=128, fov_up_deg=15.0, fov_down_deg=-15.0,
            enable_hybrid_wall_rejection=True, enable_curb_detection=True,
            curb_height_min=args.curb_min, curb_height_max=args.curb_max,
            curb_min_consecutive=2, curb_ring_gap=1,
            enable_delta_r=False, verbose=False)
        pipe_curb = LidarPipelineSuite(cfg_curb)
        r2 = pipe_curb.stage2_complete(pts)

        rgb2 = np.full(N, COLOR_GROUND, dtype=np.float32)
        rgb2[r2['obs_mask']] = COLOR_OBSTACLE
        curb_mask = r2.get('curb_mask', np.zeros(N, dtype=bool))
        rgb2[curb_mask] = COLOR_CURB

        # --- GT curb: cyan=detectado, magenta=perdido ---
        curb_pts = pts[gt_curb]
        curb_detected = r2['obs_mask'][gt_curb]
        rgb_gt = np.full(len(curb_pts), COLOR_CURB_MISS, dtype=np.float32)
        rgb_gt[curb_detected] = COLOR_CURB

        # Metricas
        g, p = gt_obs & valid, r2['obs_mask'] & valid
        tp = int(np.sum(g & p))
        fp = int(np.sum(~g & p))
        fn = int(np.sum(g & ~p))
        prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
        rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        cr = int(np.sum(gt_curb & r2['obs_mask']))
        cr_pct = cr/gt_curb.sum()*100 if gt_curb.sum()>0 else 0

        self.get_logger().info(f'  F1={f1:.1f}% P={prec:.1f}% R={rec:.1f}%')
        self.get_logger().info(f'  CurbRecall={cr_pct:.1f}% ({cr}/{gt_curb.sum()})')
        self.get_logger().info(f'  Curb detectados por Feature 1: {curb_mask.sum()}')

        # Publicar y guardar para republicar
        self._msg1 = self.create_rgb_cloud(pts, rgb1)
        self._msg2 = self.create_rgb_cloud(pts, rgb2)
        self._msg3 = self.create_rgb_cloud(curb_pts, rgb_gt)
        self.pub_stage1.publish(self._msg1)
        self.pub_curb.publish(self._msg2)
        self.pub_curb_gt.publish(self._msg3)

        self.get_logger().info('Publicado en /goose_stage1, /goose_curb, /goose_curb_gt')
        self.get_logger().info('Abre RViz y anade PointCloud2 con color por campo rgb')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curb_min', type=float, default=0.05)
    parser.add_argument('--curb_max', type=float, default=0.30)
    args = parser.parse_args()

    rclpy.init()
    node = GooseViz(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
