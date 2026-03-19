#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import argparse
import sys
from pathlib import Path
import cv2
import struct
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import colorsys
from scipy.spatial import Delaunay
import math
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time as _time
import os
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# Add current directory to path to find local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smoothing_utils import smooth_chaikin
from scipy.spatial import cKDTree

class RangeViewNode(Node):
    def __init__(self, data_path, scene, scan_start, scan_end):
        super().__init__('range_view_node')
        self.data_path = Path(data_path)
        self.scene = scene
        self.scan_start = int(scan_start)
        self.scan_end = int(scan_end)
        self.current_scan = self.scan_start
        self.frame_count = 0
        
        # Patchwork++ Initialization
        self.init_patchwork()
        
        # Dynamic Shadow Params
        self.shadow_decay_dist = 60.0 # Distance where boost drops to min
        self.shadow_min_decay = 0.2   # Minimum boost factor (20%)
        

        
        # Ranges for Depth Jump Check
        self.prev_range_image = None
        
        # Publishers
        self.range_image_pub = self.create_publisher(Image, 'range_image', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, 'point_cloud', 10)
        self.delta_r_cloud_pub = self.create_publisher(PointCloud2, 'delta_r_cloud', 10)
        self.delta_r_filtered_pub = self.create_publisher(PointCloud2, 'delta_r_filtered_cloud', 10)
        self.ground_cloud_pub = self.create_publisher(PointCloud2, 'ground_cloud', 10)
        self.pub_walls = self.create_publisher(PointCloud2, '/detected_walls', 10) # NEW: Visualize rejected walls
        self.gt_cloud_pub = self.create_publisher(PointCloud2, 'gt_cloud', 10)
        self.bayes_cloud_pub = self.create_publisher(PointCloud2, 'bayes_cloud', 10)
        self.bayes_temporal_pub = self.create_publisher(PointCloud2, 'bayes_temporal_cloud', 10)
        self.cluster_points_pub = self.create_publisher(PointCloud2, 'cluster_points', 10)
        self.shadow_pub = self.create_publisher(Marker, 'geometric_shadows', 10) # New Shadow Publisher
        self.text_pub = self.create_publisher(MarkerArray, 'shadow_text_markers', 10) # Text labels
        self.shadow_cloud_pub = self.create_publisher(PointCloud2, 'shadow_cloud', 10) # Pure Shadows
        self.pub_voids = self.create_publisher(PointCloud2, 'void_cloud', 10) # Negative Obstacles (Holes)
        
        self.void_points = []
        
        # QA Check: Best Effort for Markers to match RViz config
        # self.hull_pub = self.create_publisher(Marker, 'concave_hull', hull_qos)
        # Use simple Reliable QoS to avoid incompatibility
        self.hull_pub = self.create_publisher(Marker, 'concave_hull', 10)
        
        # Hull Attributes
        self.points_2d = None
        self.concave_hull_indices = None
        self.ALPHA_VAL = 0.1 # Tighter fit (10m radius) to match geometric_clustering
        self.MAX_RANGE = 50.0
        self.VIS_Z_LEVEL = -1.7 # Ground level for visualization
        self.detected_clusters = [] # Store clusters for shadow generation
        self.shadow_marker_msg = None # Cache for timer
        self.text_marker_array_msg = None # Cache for text
        
        # Projection Params (Velodyne HDL-64E)
        self.fov_up = 3.0 * np.pi / 180.0
        self.fov_down = -25.0 * np.pi / 180.0
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        
        self.H = 64
        self.W = 2048 # Horizontal resolution
        
        # ===== Bayesian Temporal Filter (Log-Odds) =====
        # Belief map: log-odds per cell [H, W]
        # L=0 -> P=0.5 (non-informative prior)
        self.belief_map = np.zeros((self.H, self.W), dtype=np.float64)
        self.l0 = 0.0  # Prior log-odds (P=0.5)
        self.threshold_obs = -0.3  # Unified obstacle threshold (delta_r)
        
        # Load poses for ego-motion compensation
        self.poses = self.load_poses()
        self.prev_points = None  # Points from previous frame (for association)
        
        self.get_logger().info(
            f"Bayes Filter Initialized: belief_map [{self.H}x{self.W}], "
            f"prior L0={self.l0:.2f}, poses={'loaded ('+str(len(self.poses))+')' if self.poses is not None else 'NOT FOUND'}"
        )
        
        # Default Plane (Flat ground at -1.73m)
        self.default_normal = np.array([0.0, 0.0, 1.0])
        self.default_d = 1.73
        
        # Default Plane (Flat ground at -1.73m)
        self.default_normal = np.array([0.0, 0.0, 1.0])
        self.default_d = 1.73
        
        # Standard KITTI Calibration (Tr_velo_to_cam)
        # Source: KITTI Odometry Calibration (sequence 00-21 same sensor setup usually)
        self.Tr = np.array([
            [4.2768028e-04, -9.9996725e-01, -8.0844917e-03, -1.1984599e-02],
            [-7.2106265e-03, 8.0811985e-03, -9.9994132e-01, -5.4039847e-02],
            [9.9997386e-01, 4.8594858e-04, -7.2069002e-03, -2.9219686e-01],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # State for smoothing / fallback
        self.prev_normal = self.default_normal.copy()
        self.prev_d = self.default_d
        
        self.cv_bridge = CvBridge()
        
        # Broadcaster para evitar errores de TF en RViz
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()
        
        self.get_logger().info(
            f"Range View Node Started. Scene: {scene}, "
            f"Scans: {self.scan_start:06d} -> {self.scan_end:06d} "
            f"({self.scan_end - self.scan_start + 1} frames, batch mode)"
        )
        
        # ===== Batch processing: process ALL frames upfront =====
        import time as _time
        t0 = _time.time()
        for scan_i in range(self.scan_start, self.scan_end + 1):
            self.current_scan = scan_i
            self.load_and_project()
            self.frame_count += 1
        elapsed = _time.time() - t0
        self.get_logger().info(
            f"=== Batch complete: {self.frame_count} frames in {elapsed:.1f}s ==="
        )

        # NUEVO: Guardar métricas de evaluación
        self.save_evaluation_metrics()

        # Timer only republishes the final result
        self.timer = self.create_timer(1.0, self.publish_data)

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

    def init_patchwork(self):
        try:
            import pypatchworkpp
            self.params = pypatchworkpp.Parameters()
            self.params.verbose = False
            # Config for SemanticKITTI (HDL-64E)
            self.params.sensor_height = 1.73
            self.params.num_iter = 3
            self.params.num_lpr = 20
            self.params.num_min_pts = 10
            self.params.th_dist = 0.2
            self.params.max_range = 80.0
            self.params.min_range = 2.7
            self.params.uprightness_thr = 0.707
            self.params.adaptive_seed_selection_margin = -1.1
            self.params.enable_RNR = False
            
            # Explicitly set zone parameters to match defaults (for reconstruction)
            # Default from patchworkpp.h
            self.params.num_zones = 4
            self.params.num_rings_each_zone = [2, 4, 4, 4]
            self.params.num_sectors_each_zone = [16, 32, 54, 32]

            self.patchwork = pypatchworkpp.patchworkpp(self.params)
            self.initialize_czm_params()
            self.get_logger().info("Patchwork++ Initialized Successfully")
        except ImportError:
            self.get_logger().error("Could not import pypatchworkpp. Make sure PYTHONPATH is set.")
            sys.exit(1)

    def initialize_czm_params(self):
        # Replicate CZM initialization from Patchwork++ C++ code
        min_r = self.params.min_range
        max_r = self.params.max_range
        
        self.min_ranges = [
            min_r,
            (7 * min_r + max_r) / 8.0,
            (3 * min_r + max_r) / 4.0,
            (min_r + max_r) / 2.0
        ]
        
        # Calculate ring sizes per zone
        self.ring_sizes = []
        self.ring_sizes.append((self.min_ranges[1] - self.min_ranges[0]) / self.params.num_rings_each_zone[0])
        self.ring_sizes.append((self.min_ranges[2] - self.min_ranges[1]) / self.params.num_rings_each_zone[1])
        self.ring_sizes.append((self.min_ranges[3] - self.min_ranges[2]) / self.params.num_rings_each_zone[2])
        self.ring_sizes.append((max_r - self.min_ranges[3]) / self.params.num_rings_each_zone[3])
        
        # Calculate sector sizes per zone (in radians)
        self.sector_sizes = [2 * np.pi / n for n in self.params.num_sectors_each_zone]

    def get_czm_bin(self, x, y):
        # Vectorized version for arrays
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        
        # Default invalid
        zone_idx = np.full_like(r, -1, dtype=np.int32)
        ring_idx = np.full_like(r, -1, dtype=np.int32)
        sector_idx = np.full_like(r, -1, dtype=np.int32)
        
        # Valid mask
        valid = (r > self.params.min_range) & (r <= self.params.max_range)
        
        # Iterate zones to fill indices
        # Zone 0
        mask_z0 = valid & (r < self.min_ranges[1])
        if np.any(mask_z0):
            zone_idx[mask_z0] = 0
            ring_idx[mask_z0] = ((r[mask_z0] - self.min_ranges[0]) / self.ring_sizes[0]).astype(np.int32)
            sector_idx[mask_z0] = (theta[mask_z0] / self.sector_sizes[0]).astype(np.int32)

        # Zone 1
        mask_z1 = valid & (r >= self.min_ranges[1]) & (r < self.min_ranges[2])
        if np.any(mask_z1):
            zone_idx[mask_z1] = 1
            ring_idx[mask_z1] = ((r[mask_z1] - self.min_ranges[1]) / self.ring_sizes[1]).astype(np.int32)
            sector_idx[mask_z1] = (theta[mask_z1] / self.sector_sizes[1]).astype(np.int32)

        # Zone 2
        mask_z2 = valid & (r >= self.min_ranges[2]) & (r < self.min_ranges[3])
        if np.any(mask_z2):
            zone_idx[mask_z2] = 2
            ring_idx[mask_z2] = ((r[mask_z2] - self.min_ranges[2]) / self.ring_sizes[2]).astype(np.int32)
            sector_idx[mask_z2] = (theta[mask_z2] / self.sector_sizes[2]).astype(np.int32)

        # Zone 3
        mask_z3 = valid & (r >= self.min_ranges[3])
        if np.any(mask_z3):
            zone_idx[mask_z3] = 3
            ring_idx[mask_z3] = ((r[mask_z3] - self.min_ranges[3]) / self.ring_sizes[3]).astype(np.int32)
            sector_idx[mask_z3] = (theta[mask_z3] / self.sector_sizes[3]).astype(np.int32)

        # Clip indices to be safe
        for z in range(4):
            mask = (zone_idx == z)
            if np.any(mask):
                ring_idx[mask] = np.clip(ring_idx[mask], 0, self.params.num_rings_each_zone[z] - 1)
                sector_idx[mask] = np.clip(sector_idx[mask], 0, self.params.num_sectors_each_zone[z] - 1)

        return zone_idx, ring_idx, sector_idx

    def get_czm_bin_scalar(self, x, y):
        # Scalar version for construction
        r = np.sqrt(x**2 + y**2)
        if r <= self.params.min_range or r > self.params.max_range:
            return None
            
        theta = np.arctan2(y, x)
        if theta < 0: theta += 2 * np.pi
        
        # Check zones
        if r < self.min_ranges[1]:
            z = 0
            r_base = self.min_ranges[0]
        elif r < self.min_ranges[2]:
            z = 1
            r_base = self.min_ranges[1]
        elif r < self.min_ranges[3]:
            z = 2
            r_base = self.min_ranges[2]
        else:
            z = 3
            r_base = self.min_ranges[3]
            
        r_idx = int((r - r_base) / self.ring_sizes[z])
        r_idx = min(r_idx, self.params.num_rings_each_zone[z] - 1)
        r_idx = max(r_idx, 0)
        
        s_idx = int(theta / self.sector_sizes[z])
        s_idx = min(s_idx, self.params.num_sectors_each_zone[z] - 1)
        s_idx = max(s_idx, 0)
        
        return (z, r_idx, s_idx)

    def load_and_project(self):
        # --- Usar el bin de ejemplo ---
        example_bin = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/data/000000.bin")

        if not example_bin.exists():
            self.get_logger().error(f"No se encontró el archivo de ejemplo: {example_bin}")
            return False

        bin_path = example_bin
        self.get_logger().info(f"[Frame {self.frame_count}] Loading ejemplo: {bin_path}")

        # --- Cargar poses (si existen) ---
        if hasattr(self, 'poses') and self.poses is not None and len(self.poses) > self.current_scan:
            p = self.poses[self.current_scan]
            if p.shape == (3, 4):
                self.current_pose = np.vstack([p, [0, 0, 0, 1]])
            elif p.shape == (12,):
                self.current_pose = np.vstack([p.reshape(3, 4), [0, 0, 0, 1]])
            else:
                self.current_pose = p
        else:
            self.current_pose = np.eye(4)

        # --- Cargar datos del bin ---
        scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        remissions = scan[:, 3]

        # --- Cargar etiquetas GT (opcional) ---
        label_root = self.data_path.parent.parent / 'data_odometry_labels' / 'dataset'
        scan_str = f"{int(self.current_scan):06d}"
        label_path = label_root / 'sequences' / self.scene / 'labels' / f"{scan_str}.label"

        self.has_gt = False
        if label_path.exists():
            labels_raw = np.fromfile(label_path, dtype=np.uint32)
            self.gt_semantic = labels_raw & 0xFFFF
            self.has_gt = True
            self.get_logger().info(f"Loaded GT labels: {len(self.gt_semantic)} points, {len(np.unique(self.gt_semantic))} classes")
        else:
            self.get_logger().warn(f"No GT labels found at {label_path}")

        t_start = _time.time()
        
        # --- 1. Ground Segmentation (Patchwork++) ---
        self.patchwork.estimateGround(points)
        ground_points = self.patchwork.getGround()
        
        # Get Local Planes
        # centers: (N_planes, 3) - Centroids of fitted patches
        # normals: (N_planes, 3) - Normals of fitted patches
        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()
        
        # Build Lookup Table: (zone, ring, sector) -> (normal, d)
        # FILTER: Only accept planes with normal pointing UP (n_z > 0.7)
        # This rejects vertical surfaces (walls) whose normals are horizontal.
        local_planes = {}
        rejected_bins = set()  # Track bins with non-horizontal planes (walls)
        n_rejected = 0
        
        rejected_centroids = [] # Store centroids of rejected planes for visualization

        if len(centers) > 0:
            # Build KDTree for precise slope analysis
            try:
                # ... (KDTree logic remains)

                ground_tree = cKDTree(ground_points)
                has_tree = True
            except Exception as e:
                self.get_logger().warn(f"KDTree Failed: {e}")
                has_tree = False

            for i in range(len(centers)):
                c = centers[i]
                n = normals[i]
                
                bin_id = self.get_czm_bin_scalar(c[0], c[1])
                
                # SLOPE-AWARE WALL REJECTION (SOTA)
                # 1. Check Normal: If horizontal (abs(n[2]) < 0.7), SUSPECT WALL.
                # 2. Check Height Variation (Delta Z):
                #    - If Delta Z < 0.3m (30cm), it's a CURB or RAMP -> ACCEPT.
                #    - If Delta Z > 0.3m, it's a WALL -> REJECT.
                
                is_wall = False
                if abs(n[2]) < 0.7:
                    # Validate with local geometry
                    if has_tree:
                        # Find points in 0.5m radius (more local)
                        idx = ground_tree.query_ball_point(c, r=0.5)
                        if len(idx) > 5: # Need enough points for percentile
                            local_pts = ground_points[idx]
                            # Robust Delta Z using percentiles (ignores outliers)
                            z_high = np.percentile(local_pts[:, 2], 95)
                            z_low = np.percentile(local_pts[:, 2], 5)
                            delta_z = z_high - z_low
                            
                            # If significant height difference, confirmed WALL
                            if delta_z > 0.3:
                                is_wall = True
                                # self.get_logger().info(f"Wall DETECTED: nz={n[2]:.3f} dZ={delta_z:.3f} -> REJECT") 
                            else:
                                # Small height diff -> CURB/RAMP -> ACCEPT
                                is_wall = False
                                # self.get_logger().info(f"Wall INDULTADO (Ramp): nz={n[2]:.3f} dZ={delta_z:.3f} -> ACCEPT")
                        else:
                            # Not enough points to judge, rely on absolute height heuristic
                            if c[2] > -1.0: 
                                is_wall = True
                                # self.get_logger().info(f"Wall HEURISTIC (High): Z={c[2]:.2f} -> REJECT")
                    else:
                        # Fallback if tree failed
                        if c[2] > -1.0: is_wall = True
                
                if is_wall:
                    n_rejected += 1
                    # Visualize logic: Store centroid
                    rejected_centroids.append(c)
                    if bin_id is not None:
                        rejected_bins.add(bin_id)
                    continue
                
                # Ensure normal points upward
                if n[2] < 0:
                    n = -n
                
                # Determine ID
                if bin_id is not None:
                    # d = -n . c
                    d = -np.dot(n, c)
                    local_planes[bin_id] = (n, d)
        
        self.rejected_bins = rejected_bins
        self.get_logger().info(f"Generated {len(local_planes)} local ground planes ({n_rejected} rejected, {len(rejected_bins)} wall bins).")

        # Also keep a Global Plane as fallback
        global_normal = self.prev_normal
        global_d = self.prev_d
        
        if len(ground_points) > 10:
            centroid = np.mean(ground_points, axis=0)
            centered_ground = ground_points - centroid
            U, S, Vt = np.linalg.svd(centered_ground, full_matrices=False)
            candidate_normal = Vt[2, :]
            
            if candidate_normal[2] < 0:
                candidate_normal = -candidate_normal
                
            candidate_d = -np.dot(candidate_normal, centroid)
            
            if candidate_normal[2] > 0.9:
                global_normal = candidate_normal
                global_d = candidate_d
                self.prev_normal = global_normal
                self.prev_d = global_d
        
        self.ground_cloud_msg = self.create_cloud(ground_points, np.zeros(len(ground_points)))
        
        # --- PROJECTION with Local Planes ---
        
        # 1. Bin all points to find their corresponding plane
        # Helper returns arrays of indices
        z_idx, r_idx, s_idx = self.get_czm_bin(points[:, 0], points[:, 1])
        
        # 2. Calculate Expected Range (r_exp) per point
        r = np.linalg.norm(points, axis=1)
        
        # Initialize with fallback (global plane)
        r_exp = np.full_like(r, 999.9)
        
        # Default Fallback Calculation (Global)
        dot_global = np.dot(points, global_normal)
        valid_global = np.abs(dot_global) > 1e-3
        
        # To avoid division by zero
        dot_global[~valid_global] = 1e-3 
        
        # --- VECTORIZED PLANE LOOKUP ---
        # Create Global Plane Table (Zone, Ring, Sector, 4) -> [nx, ny, nz, d]
        # Max dimensions based on CZM config: 4 zones, max 4 rings, max 54 sectors
        # Initialize with FALLBACK (Global Plane)
        planes_table = np.zeros((4, 4, 54, 4), dtype=np.float32)
        planes_table[..., :3] = global_normal
        planes_table[..., 3] = global_d
        
        # Fill Table with Local Planes
        for bin_id, (n_loc, d_loc) in local_planes.items():
            z_b, r_b, s_b = bin_id
            # Safe guard indices
            if 0 <= z_b < 4 and 0 <= r_b < 4 and 0 <= s_b < 54:
                planes_table[z_b, r_b, s_b, :3] = n_loc
                planes_table[z_b, r_b, s_b, 3] = d_loc
                
        # Lookup Normals and D for all points
        # Handle invalid indices (-1) by clipping to 0 (will use fallback/garbage but filtered later)
        # or better: use a mask for valid bins
        valid_bins = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)
        
        # Create arrays for n and d per point
        n_per_point = np.full((len(points), 3), global_normal, dtype=np.float32)
        d_per_point = np.full(len(points), global_d, dtype=np.float32)
        
        if np.any(valid_bins):
            # Advanced Indexing
            # We need to ensure indices are within bounds. z_idx is 0-3. r_idx max 3. s_idx max 53.
            # Only access valid ones.
            z_v = z_idx[valid_bins]
            r_v = r_idx[valid_bins]
            s_v = s_idx[valid_bins]
            
            # Retrieve from table
            plane_params = planes_table[z_v, r_v, s_v] # (N_valid, 4)
            
            n_per_point[valid_bins] = plane_params[:, :3]
            d_per_point[valid_bins] = plane_params[:, 3]
            
        # Calculate r_exp vectorized
        # r_exp = -d / (n . p_normalized * r) ? No, n . p = -d/r
        # r_exp = -d / (n . (p/r))
        # Wait, simple formula: P_exp on plane satisfies n . P_exp + d = 0
        # P_exp = r_exp * (P_measured / r_measured)  (Ray casting approximation)
        # n . (r_exp * P/r) + d = 0
        # r_exp * (n . P)/r = -d
        # r_exp = -d * r / (n . P)
        
        dot_prod = np.sum(points * n_per_point, axis=1)
        
        # Avoid division by zero (parallel rays)
        valid_dot = dot_prod < -1e-3 # Only downward facing
        
        # Initialize r_exp with large value (sky)
        r_exp = np.full(len(points), 999.9, dtype=np.float32)
        
        # Valid intersections
        r_exp[valid_dot] = -d_per_point[valid_dot] * r[valid_dot] / dot_prod[valid_dot]

        # Delta R
        delta_r = r - r_exp
        
        # Filter extremes
        delta_r = np.clip(delta_r, -20.0, 10.0)
        
        t_proj = _time.time()
        # --- Mapping to Image ---
        # 2. Key: Get vertical angle (Pitch) -> Ring ID
        pitch = np.arcsin(points[:, 2] / r)
        
        # 3. Get horizontal angle (Yaw) -> Column
        yaw = np.arctan2(points[:, 1], points[:, 0])
        
        # Row (u)
        proj_y = (pitch + abs(self.fov_down)) / self.fov # 0..1
        proj_y = 1.0 - proj_y # 1..0 (Top is 0)
        u = np.floor(proj_y * self.H).astype(np.int32)
        u = np.clip(u, 0, self.H - 1)
        
        # Col (v)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        v = np.floor(proj_x * self.W).astype(np.int32)
        v = np.clip(v, 0, self.W - 1)
        
        # Create Image
        self.range_image = np.full((self.H, self.W), 0.0, dtype=np.float32)
        
        # Order by range (descending) so last write (closest) wins
        order = np.argsort(r)[::-1]
        
        u_sorted = u[order]
        v_sorted = v[order]
        delta_sorted = delta_r[order]
        r_exp_sorted = r_exp[order]
        
        # Valid mask for visualization:
        # If r_exp is not sky-like
        
        self.range_image[u_sorted, v_sorted] = delta_sorted
        
        # --- Build rejected-bin mask ---
        # Points in bins whose planes were rejected (vertical walls)
        # These should be forced to obstacle regardless of delta_r
        self.rejected_mask = np.zeros(len(points), dtype=bool)
        for bin_id in self.rejected_bins:
            z_bin, r_bin, s_bin = bin_id
            mask = (z_idx == z_bin) & (r_idx == r_bin) & (s_idx == s_bin)
            self.rejected_mask |= mask
        
        n_rejected_pts = np.sum(self.rejected_mask)
        self.get_logger().info(f"Points in rejected (wall) bins: {n_rejected_pts}")
        
        # VISUALIZATION: Publish Detected Walls (Red CENTROIDS GLOBAL)
        if len(rejected_centroids) > 0:
            local_pts = np.array(rejected_centroids)
            
            # Transform to Global Map Frame
            # Pose from KITTI is 3x4 usually.
            pose = self.current_pose
            if pose.shape == (3, 4):
                 pose = np.vstack([pose, [0, 0, 0, 1]])
            
            # hom = np.hstack([local_pts, np.ones((len(local_pts), 1))])
            # global_pts = (pose @ hom.T).T[:, :3]
            global_pts = local_pts # Local
            
            rgb_packed = np.full(len(global_pts), 0x00FF0000, dtype=np.uint32)
            # Create cloud
            self.wall_msg = self.create_cloud(global_pts, rgb_packed.view(np.float32), field_name='rgb')
            self.wall_msg.header.frame_id = "velodyne" 
            # self.pub_walls.publish(self.wall_msg)
        else:
            dummy_pts = np.zeros((0, 3), dtype=np.float32)
            dummy_rgb = np.zeros(0, dtype=np.float32)
            self.wall_msg = self.create_cloud(dummy_pts, dummy_rgb, field_name='rgb')
            # self.pub_walls.publish(self.wall_msg)
        
        # Stats
        valid_delta = delta_sorted[np.abs(delta_sorted) < 20.0]
        if len(valid_delta) > 0:
            self.get_logger().info(f"Delta R Stats: Min={np.min(valid_delta):.3f}, Max={np.max(valid_delta):.3f}, Mean={np.mean(valid_delta):.3f}")
        else:
             self.get_logger().info(f"Delta R Stats: All invalid/sky")

        self.get_logger().info("Projection Complete")
        
        # Prepare Delta R PointCloud2 message
        # Calculate Colors:
        # Obstacle (Delta < -0.2) -> Blue (0, 0, 255)
        # Ground (-0.2 <= Delta <= 0.2) -> Grey (128, 128, 128)
        # Depression (0.2 < Delta < 2.0) -> Orange (255, 165, 0)
        # Sky/Unknown (Delta >= 2.0) -> Cyan (0, 255, 255)
        
        # Vectorized Color Generation
        # Default Grey (Ground)
        r_c = np.full(delta_r.shape, 128, dtype=np.uint32)
        g_c = np.full(delta_r.shape, 128, dtype=np.uint32)
        b_c = np.full(delta_r.shape, 128, dtype=np.uint32)
        
        # --- Common masks ---
        mask_obs_raw = delta_r < self.threshold_obs  # Raw obstacle (unified threshold)
        mask_dep = (delta_r > 0.2) & (delta_r < 2.0)
        mask_sky = delta_r >= 2.0
        
        # Smoothed obstacle mask (geometric inter-ring consistency)
        mask_obs_2d = self.apply_geometric_consistency(self.range_image)
        mask_obs_smooth = mask_obs_2d[u, v]
        
        # Combined raw obstacle = threshold OR wall-bin override
        mask_obs_combined = mask_obs_raw | self.rejected_mask
        
        # ========================================
        # CLOUD 1: PRE-FILTER (raw, no inter-ring)
        # ========================================
        r_c[mask_obs_combined] = 0
        g_c[mask_obs_combined] = 0
        b_c[mask_obs_combined] = 255
        
        # Depression and Sky should NOT override obstacles/walls
        mask_dep_clean = mask_dep & ~mask_obs_combined
        mask_sky_clean = mask_sky & ~mask_obs_combined
        
        r_c[mask_dep_clean] = 255; g_c[mask_dep_clean] = 165; b_c[mask_dep_clean] = 0
        r_c[mask_sky_clean] = 0;   g_c[mask_sky_clean] = 255; b_c[mask_sky_clean] = 255
        
        rgb_pre = (r_c << 16) | (g_c << 8) | b_c
        self.delta_r_msg = self.create_cloud(points, rgb_pre.view(np.float32), field_name='rgb')
        
        # =========================================
        # CLOUD 2: POST-FILTER (with diff colors)
        # =========================================
        # Start from same base as pre-filter
        r_f = r_c.copy()
        g_f = g_c.copy()
        b_f = b_c.copy()
        
        # Diff masks (relative to raw+wall combined)
        # Points REMOVED by filter: were obstacle in raw but NOT after smoothing
        mask_removed = mask_obs_combined & ~mask_obs_smooth & ~self.rejected_mask
        # Points ADDED by filter: NOT obstacle in raw but ARE after smoothing
        mask_added = ~mask_obs_combined & mask_obs_smooth
        
        # Red = removed (filter says it was noise)
        r_f[mask_removed] = 255
        g_f[mask_removed] = 0
        b_f[mask_removed] = 0
        
        # Green = added (filter filled a gap)
        r_f[mask_added] = 0
        g_f[mask_added] = 255
        b_f[mask_added] = 0
        
        rgb_post = (r_f << 16) | (g_f << 8) | b_f
        self.delta_r_filtered_msg = self.create_cloud(points, rgb_post.view(np.float32), field_name='rgb')
        
        # Log filter stats
        n_obs_raw = np.sum(mask_obs_combined)
        n_removed = np.sum(mask_removed)
        n_added = np.sum(mask_added)
        n_wall = np.sum(self.rejected_mask)
        self.get_logger().info(
            f"Filter Stats: {n_obs_raw} raw obstacles ({n_wall} wall), "
            f"{n_removed} removed (Red), {n_added} added (Green)"
        )
        
        # =============================================
        # CLOUD 4: BAYESIAN TEMPORAL FILTER (Log-Odds)
        # =============================================
        # NEW PIPELINE ORDER: Raw -> Bayes -> Spatial Smoothing
        
        # Store Range Image for Shadow Logic
        # Create a range image from the current points for shadow casting
        r_range_view = np.full((self.H, self.W), np.inf, dtype=np.float32)
        r_range_view[u, v] = r
        self.current_range_image = r_range_view.copy()

        # 1. Raw Probability from current frame
        P_raw_2d = self.get_raw_probability(self.range_image)
        
        # 2. Bayes Filter (Temporal Consistency)
        # Updates belief map with raw observation
        P_belief, self.belief_map = self.update_belief(self.belief_map, P_raw_2d, points)
        
        t_bayes = _time.time()
        
        # --- SHADOW BOOST (Geometric Validation) ---
        # Apply boost BEFORE spatial smoothing so it propagates to neighbors
        # We need u_all, v_all for shadow map indexing
        u_all, v_all, _ = self.project_points_to_uv(points)
        
        # MUST pass Absolute Range (current_range_image), not Delta R
        # Optimized: Pass d_per_point to avoid re-lookup
        shadow_boost = self.detect_geometric_shadows(self.current_range_image, u_all, v_all, points, d_per_point)
        self.belief_map += shadow_boost
        
        # Publish Void Cloud (Negative Obstacles)
        if len(self.void_points) > 0:
            p_void = np.array(self.void_points)
            # Transform to Map
            # hom = np.hstack([p_void, np.ones((len(p_void), 1))])
            # global_void = (pose @ hom.T).T[:, :3]
            global_void = p_void # Local
            
            # Violet Color for Voids (Holes)
            rgb_packed = np.full(len(global_void), 0x00800080, dtype=np.uint32)
            
            self.void_msg = self.create_cloud(global_void, rgb_packed.view(np.float32), field_name='rgb')
            self.void_msg.header.frame_id = "velodyne"
            # self.pub_voids.publish(self.void_msg)
        else:
            self.void_msg = None
            
        # Store current range image for next frame's Jump Check
        self.prev_abs_range = self.current_range_image.copy()
        
        t_shadow = _time.time()
        
        # Re-clamp and Re-calc P_belief after boost
        np.clip(self.belief_map, -5.0, 5.0, out=self.belief_map)
        P_belief = 1.0 / (1.0 + np.exp(-self.belief_map))
        
        # Publish Intermediate Temporal Cloud (With Shadow Boost, Pre-Smooth)
        P_belief_per_point = P_belief[u, v]
        rgb_temporal = self.probabilities_to_rgb(P_belief_per_point, points)
        self.bayes_temporal_msg = self.create_cloud(points, rgb_temporal.view(np.float32), field_name='rgb')
        
        # 3. Spatial Smoothing (Inter-ring Consistency) on BOOSTED belief
        P_final = self.apply_spatial_smoothing(P_belief)

        # Guardar para evaluación offline
        self.belief_prob = P_final  # (H, W) probabilidad final después de smoothing

        t_smooth = _time.time()
        
        # Map final probability to per-point colors
        # Lookup belief for each point using its (u, v) projection
        P_per_point = P_final[u, v]
        
        # Map final probability to colors
        rgb_bayes = self.probabilities_to_rgb(P_per_point, points)
        self.bayes_cloud_msg = self.create_cloud(points, rgb_bayes.view(np.float32), field_name='rgb')
        
        
        # 6. Cluster Objects (Using P_final)
        # We pass u_all, v_all to map back to pixels if needed
        u_all, v_all, _ = self.project_points_to_uv(points) # Re-project to get indices for filtered points
        
        # Call Clustering (populates self.detected_clusters and returns colored cloud)
        self.cluster_points_msg = self.cluster_objects(points, P_per_point, u_all, v_all)
        
        t_cluster = _time.time()
        
        # --- HULL UPDATE (Using ALL points to define free space) ---
        if len(points) > 100:
            self.compute_concave_hull(points)
        
        # 9. Generate Geometric Shadows (Requires Clusters + Hull)
        if self.detected_clusters and (self.concave_hull_indices is not None and len(self.concave_hull_indices) > 0):
            shadow_marker = self.generate_geometric_shadows(points) # Pass all points for context
            if shadow_marker:
                # self.shadow_pub.publish(shadow_marker)
                self.shadow_marker_msg = shadow_marker
        
        # Bayes stats (recalculated from P_per_point for simplicity or reuse masks)
        # Confirmed obstacle (P > 0.8)
        n_conf_obs = np.sum(P_per_point > 0.8)
        n_prob_obs = np.sum((P_per_point > 0.6) & (P_per_point <= 0.8))
        n_indet = np.sum((P_per_point > 0.4) & (P_per_point <= 0.6))
        n_prob_gnd = np.sum((P_per_point >= 0.2) & (P_per_point <= 0.4))
        n_conf_gnd = np.sum(P_per_point < 0.2)
        
        self.get_logger().info(
            f"Bayes Stats [Frame {self.frame_count}]: "
            f"Confirmed_Obs={n_conf_obs}, Probable_Obs={n_prob_obs}, "
            f"Indet={n_indet}, Probable_Gnd={n_prob_gnd}, Confirmed_Gnd={n_conf_gnd}"
        )
        
        # ========================================
        # CLOUD 3: GROUND TRUTH (SemanticKITTI)

        if self.has_gt and len(self.gt_semantic) == len(points):
            # ---- Per-class color mapping (R, G, B) ----
            # Each SemanticKITTI class gets a unique, visually distinct color.
            LABEL_COLORS = {
                # --- Unlabeled / Outlier ---
                0:   (50,  50,  50),   # unlabeled       -> dark grey
                1:   (70,  70,  70),   # outlier         -> slightly lighter grey

                # --- Ground sub-classes (key for surface analysis) ---
                40:  (128,  0, 255),   # road            -> purple
                44:  (255, 120,  0),   # parking         -> orange
                48:  (255,  0, 200),   # sidewalk        -> pink/magenta
                49:  (139,  69,  19),  # other-ground    -> brown (curbs, ramps, etc.)
                60:  (255, 255, 255),  # lane-marking    -> white
                72:  (210, 190, 100),  # terrain         -> sandy yellow

                # --- Vehicles ---
                10:  (0,   0,  255),   # car             -> blue
                11:  (120, 0,  255),   # bicycle         -> violet
                13:  (0,  60,  200),   # bus             -> dark blue
                15:  (0, 100,  255),   # motorcycle      -> medium blue
                16:  (100, 100, 200),  # on-rails       -> steel blue
                18:  (0,  30,  180),   # truck           -> navy
                20:  (80, 80,  255),   # other-vehicle   -> light blue

                # --- Humans ---
                30:  (255, 255,   0),  # person          -> yellow
                31:  (200, 255,   0),  # bicyclist       -> yellow-green
                32:  (255, 200,   0),  # motorcyclist    -> amber

                # --- Structures ---
                50:  (180,   0,   0),  # building        -> dark red
                51:  (0,  200, 200),   # fence           -> cyan
                52:  (200, 100,  50),  # other-structure -> copper

                # --- Vegetation ---
                70:  (0,  180,   0),   # vegetation      -> green
                71:  (100, 60,   0),   # trunk           -> dark brown

                # --- Urban furniture ---
                80:  (220,   0, 220),  # pole            -> magenta
                81:  (220, 180,   0),  # traffic-sign    -> gold
                99:  (150, 150, 150),  # other-object    -> light grey

                # --- Moving objects ---
                252: (50,  50,  255),  # moving-car           -> bright blue
                253: (200, 255,  50),  # moving-bicyclist     -> lime
                254: (255, 255, 100),  # moving-person        -> light yellow
                255: (255, 180,  50),  # moving-motorcyclist  -> light amber
                256: (120, 120, 220),  # moving-on-rails      -> lavender
                257: (30,  80,  220),  # moving-bus           -> medium blue
                258: (30,  50,  200),  # moving-truck         -> indigo
                259: (100, 100, 255),  # moving-other-vehicle -> periwinkle
            }

            # Log legend ONLY ONCE
            if self.frame_count == 0:
                self.get_logger().info("=== GT_cloud Color Legend ===")
                legend_groups = {
                    'GROUND':      [40, 44, 48, 49, 60, 72],
                    'VEHICLES':    [10, 11, 13, 15, 16, 18, 20],
                    'HUMANS':      [30, 31, 32],
                    'STRUCTURES':  [50, 51, 52],
                    'VEGETATION':  [70, 71],
                    'FURNITURE':   [80, 81, 99],
                }
                label_names = {
                    0:'unlabeled', 1:'outlier',
                    10:'car', 11:'bicycle', 13:'bus', 15:'motorcycle',
                    16:'on-rails', 18:'truck', 20:'other-vehicle',
                    30:'person', 31:'bicyclist', 32:'motorcyclist',
                    40:'road', 44:'parking', 48:'sidewalk', 49:'other-ground',
                    60:'lane-marking', 70:'vegetation', 71:'trunk', 72:'terrain',
                    50:'building', 51:'fence', 52:'other-structure',
                    80:'pole', 81:'traffic-sign', 99:'other-object',
                    252:'moving-car', 253:'moving-bicyclist', 254:'moving-person',
                    255:'moving-motorcyclist', 256:'moving-on-rails',
                    257:'moving-bus', 258:'moving-truck', 259:'moving-other-vehicle',
                }
                for grp, ids in legend_groups.items():
                    # Filter existing keys to avoid KeyError strictness in log (optional)
                    items = [f"{label_names.get(i, '?')}({i})=RGB{LABEL_COLORS.get(i, '?')}" for i in ids if i in LABEL_COLORS]
                    self.get_logger().info(f"  {grp}: {', '.join(items)}")

            # Optimized: Vectorized Color Lookup (LUT)
            # Create LUT (Max Label ~300)
            max_label = 300
            lut_r = np.full(max_label, 50, dtype=np.uint32)
            lut_g = np.full(max_label, 50, dtype=np.uint32)
            lut_b = np.full(max_label, 50, dtype=np.uint32)
            
            for lbl, (r, g, b) in LABEL_COLORS.items():
                if lbl < max_label:
                    lut_r[lbl] = r
                    lut_g[lbl] = g
                    lut_b[lbl] = b
            
            # Clip indices to avoid crash
            safe_labels = np.clip(self.gt_semantic, 0, max_label - 1)
            
            r_gt = lut_r[safe_labels]
            g_gt = lut_g[safe_labels]
            b_gt = lut_b[safe_labels]

            rgb_gt = (r_gt << 16) | (g_gt << 8) | b_gt
            self.gt_cloud_msg = self.create_cloud(points, rgb_gt.view(np.float32), field_name='rgb')
            
            # --- Accuracy Metrics (3 methods) ---
            GROUND_IDS = [40, 44, 48, 49, 60, 72]
            OBSTACLE_IDS = [10, 11, 13, 15, 16, 18, 20,   # vehicles
                            30, 31, 32,                     # humans
                            50, 51, 52,                     # structures
                            70, 71,                         # vegetation
                            80, 81, 99,                     # furniture
                            252, 253, 254, 255, 256, 257, 258, 259]  # moving
            gt_is_obstacle = np.isin(self.gt_semantic, OBSTACLE_IDS)
            gt_is_ground = np.isin(self.gt_semantic, GROUND_IDS)
            labeled = ~np.isin(self.gt_semantic, [0, 1])
            
            def calc_metrics(pred_mask, label):
                TP = int(np.sum(pred_mask & gt_is_obstacle & labeled))
                FP = int(np.sum(pred_mask & gt_is_ground & labeled))
                FN = int(np.sum(~pred_mask & gt_is_obstacle & labeled))
                TN = int(np.sum(~pred_mask & gt_is_ground & labeled))
                prec = TP / (TP + FP) if (TP + FP) > 0 else 0
                rec = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
                self.get_logger().info(
                    f"GT [{label}]: TP={TP}, FP={FP}, FN={FN}, TN={TN} | "
                    f"P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}"
                )
            
            # Method 3: Bayes Temporal (Temporal Only)
            bayes_temp_mask = P_belief_per_point > 0.5
            calc_metrics(bayes_temp_mask, "Bayes_Temp")
            
            # Method 4: Bayes Final (Temporal + Spatial)
            bayes_final_mask = P_per_point > 0.5
            calc_metrics(bayes_final_mask, "Bayes_Final")

        t_end = _time.time()
        
        self.get_logger().info(
             f"Timing [Frame {self.current_scan}]: "
             f"Proj: {(t_proj - t_start)*1000:.1f}ms, "
             f"Bayes: {(t_bayes - t_proj)*1000:.1f}ms, "
             f"Shadow: {(t_shadow - t_bayes)*1000:.1f}ms, "
             f"Smooth: {(t_smooth - t_shadow)*1000:.1f}ms, "
             f"Cluster: {(t_cluster - t_smooth)*1000:.1f}ms, "
             f"Total: {(t_end - t_start)*1000:.1f}ms"
        )

    def apply_geometric_consistency(self, range_image):
        """
        Applies geometric inter-ring regularization (smoothing) to enforce consistency.
        Based on "Super-Resolution LiDAR" logic.
        Formula: P_smooth = (1 - mu) * P + mu * (P_prev + P_next) / 2
        
        Args:
            range_image (np.array): 2D range image (H, W).
            
        Returns:
            mask_consistent (np.array): Boolean mask of consistent obstacles.
        """
        P_smooth = self.apply_geometric_consistency_prob(range_image)
        threshold_consistency = 0.6
        mask_consistent = P_smooth > threshold_consistency
        return mask_consistent

    def get_raw_probability(self, range_image):
        """Returns raw probability map derived directly from delta_r."""
        return (range_image < self.threshold_obs).astype(np.float32)

    def apply_spatial_smoothing(self, P_input):
        """
        Applies horizontal spatial smoothing (inter-ring consistency).
        Input P can be from raw measurement OR from Bayes belief map.
        
        Args:
            P_input (np.array): Probability map [0, 1] per cell.
            
        Returns:
            P_smooth (np.array): Smoothed probability [0, 1].
        """
        mu = 0.5
        
        # Circular Convolution (wrap-around)
        P_left = np.roll(P_input, 1, axis=1)
        P_right = np.roll(P_input, -1, axis=1)
        
        # Regularization
        current_term = (1.0 - mu) * P_input
        neighbor_term = mu * (P_left + P_right) / 2.0
        
        P_smooth = current_term + neighbor_term
        return P_smooth
        
    def apply_geometric_consistency_prob(self, range_image):
        """
        Legacy wrapper: returns smoothed probability from range image.
        Used by the raw mask logic.
        """
        P_raw = self.get_raw_probability(range_image)
        return self.apply_spatial_smoothing(P_raw)

    def load_poses(self):
        """
        Load poses.txt from SemanticKITTI dataset.
        Each line is a 3x4 transformation matrix (row-major): world <- lidar.
        Returns list of 4x4 numpy matrices, or None if not found.
        """
        # Try multiple possible locations
        candidates = [
            self.data_path / 'sequences' / self.scene / 'poses.txt',
            self.data_path.parent.parent / 'data_odometry_labels' / 'dataset' / 'sequences' / self.scene / 'poses.txt',
        ]
        
        pose_path = None
        for c in candidates:
            if c.exists():
                pose_path = c
               
        # Range Image Storage
        self.current_range_image = None # To store (H, W) ranges for shadow check
        if pose_path is None:
            self.get_logger().warn(f"poses.txt not found. Tried: {candidates}")
            return None
        
        self.get_logger().info(f"Loading poses from {pose_path}")
        poses = []
        with open(pose_path, 'r') as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                T = np.eye(4)
                T[:3, :] = np.array(values).reshape(3, 4)
                poses.append(T)
        
        return poses

    def project_points_to_uv(self, points):
        """
        Project 3D points to range image (u, v) coordinates.
        Same math as in load_and_project but vectorized and standalone.
        
        Returns:
            u, v: integer arrays of image coordinates
            valid: boolean mask of points within FOV
        """
        r = np.linalg.norm(points, axis=1)
        valid = r > 0.1  # Avoid division by zero
        
        pitch = np.zeros_like(r)
        pitch[valid] = np.arcsin(np.clip(points[valid, 2] / r[valid], -1, 1))
        yaw = np.arctan2(points[:, 1], points[:, 0])
        
        proj_y = (pitch + abs(self.fov_down)) / self.fov
        proj_y = 1.0 - proj_y
        u = np.floor(proj_y * self.H).astype(np.int32)
        u = np.clip(u, 0, self.H - 1)
        
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        v = np.floor(proj_x * self.W).astype(np.int32)
        v = np.clip(v, 0, self.W - 1)
        
        # Mark out-of-FOV as invalid
        valid &= (pitch >= self.fov_down) & (pitch <= self.fov_up)
        
        return u, v, valid

    def warp_belief_map(self, belief_map, current_scan_idx, points_current):
        """
        Warp the previous belief map into the current frame's coordinates
        using the relative pose between consecutive frames.
        """
        prev_scan_idx = current_scan_idx - 1
        
        # Check if we have poses and a previous frame
        if (self.poses is None or 
            prev_scan_idx < 0 or
            current_scan_idx >= len(self.poses) or
            prev_scan_idx >= len(self.poses)):
            # No warp possible - first frame or no poses
            return belief_map
        
        # Relative transform: T_prev <- T_curr
        # T_world_curr: world <- curr
        # T_world_prev: world <- prev  
        # T_prev_curr = inv(T_world_prev) @ T_world_curr  (prev <- curr)
        T_world_curr = self.poses[current_scan_idx]
        T_world_prev = self.poses[prev_scan_idx]
        
        # Compute Relative Motion in CAMERA Frame
        T_rel_cam = np.linalg.inv(T_world_prev) @ T_world_curr
        
        # Convert to LIDAR Frame: T_lidar = inv(Tr) @ T_cam @ Tr
        T_prev_curr = np.linalg.inv(self.Tr) @ T_rel_cam @ self.Tr
        
        # Transform current points into previous frame
        N = points_current.shape[0]
        pts_hom = np.hstack([points_current, np.ones((N, 1))])  # (N, 4)
        pts_in_prev = (T_prev_curr @ pts_hom.T).T[:, :3]  # (N, 3)
        
        # Project into previous frame's range image
        u_prev, v_prev, valid = self.project_points_to_uv(pts_in_prev)
        
        # Create warped belief: for each current point, look up previous belief
        warped_belief = np.zeros((self.H, self.W), dtype=np.float64)
        
        # Build current frame's (u, v) projection
        u_curr, v_curr, _ = self.project_points_to_uv(points_current)
        
        # For valid points, transfer previous belief to current position
        # For valid points, transfer previous belief to current position
        if np.any(valid):
            # --- DEPTH JUMP CHECK (Data Association) ---
            # Don't inherit belief if the point lands on a background/foreground discontinuity
            mask_depth_associ = np.ones(valid.shape, dtype=bool)
            
            if self.prev_abs_range is not None:
                # Get range seen by sensor in previous frame at projected pixels
                r_sensor_prev = self.prev_abs_range[u_prev[valid], v_prev[valid]]
                
                # Get range of the current point in previous frame
                r_point_prev = np.linalg.norm(pts_in_prev[valid], axis=1)
                
                # Diff check: if > 0.2m (was 0.5), likely hitting different object -> reset belief
                # Stricter check aids in separating wall base from ground
                diff = np.abs(r_sensor_prev - r_point_prev)
                mask_depth_associ = diff < 0.2
                
            # --- WALL RESET (Geometry Priority) ---
            # If the point is geometrically a wall (rejected by Patchwork), DO NOT inherit ground history.
            # We force a reset so the strong current obstacle likelihood takes over immediately.
            mask_not_wall = np.ones(valid.shape, dtype=bool)
            if hasattr(self, 'rejected_mask'):
                 # rejected_mask is aligned with 'points' (N,)
                 # We need to subset it for 'valid' points
                 # valid is a boolean mask of shape (N,) from project_points_to_uv?
                 # No, valid is (N,) boolean from project_points_to_uv
                 mask_not_wall = ~self.rejected_mask[valid]
            
            # Combine masks
            final_mask = valid.copy()
            final_mask[valid] &= (mask_depth_associ & mask_not_wall)
            
            if np.any(final_mask):
                prev_beliefs = belief_map[u_prev[final_mask], v_prev[final_mask]]
                # Use the current frame's (u,v) position
                np.add.at(warped_belief, (u_curr[final_mask], v_curr[final_mask]), prev_beliefs)
                count = np.zeros((self.H, self.W), dtype=np.float64)
                np.add.at(count, (u_curr[final_mask], v_curr[final_mask]), 1.0)
                nonzero = count > 0
                warped_belief[nonzero] /= count[nonzero]
        
        return warped_belief

    def update_belief(self, belief_map_state, P_obs, points_current):
        """
        Bayesian temporal update using log-odds representation.
        Refactored to be stateless (returns new state).
        
        Args:
            belief_map_state (np.array): [H, W] current accumulated belief state.
            P_obs (np.array): [H, W] observation probability from current frame.
            points_current (np.array): [N, 3] current frame's 3D points.
            
        Returns:
            P_belief (np.array): [H, W] updated belief probability.
            new_belief_map (np.array): [H, W] updated belief state (log-odds).
        """
        # Step 1: Warp previous belief to current frame coordinates
        warped_belief = self.warp_belief_map(belief_map_state, self.current_scan, points_current)
        
        # Step 2: Convert current observation to log-odds
        eps = 1e-6
        P_clamped = np.clip(P_obs, eps, 1.0 - eps)
        l_obs = np.log(P_clamped / (1.0 - P_clamped))
        
        # Step 3: Recursive Bayes update with warped prior
        # L_t = L_{t-1}^{warped} + l_obs - l_0
        new_belief_map = warped_belief + l_obs - self.l0
        
        # Clamp to avoid excessive inertia (-2.5 to 2.5 -> P=0.07 to 0.93)
        # Was 5.0, reducing to 2.5 makes it very reactive (less memory)
        new_belief_map = np.clip(new_belief_map, -2.5, 2.5)
        
        # Convert back to probability for visualization
        P_belief = 1.0 / (1.0 + np.exp(-new_belief_map))
        
        return P_belief, new_belief_map


    def create_cloud(self, points, values, field_name='intensity'):
        msg = PointCloud2()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # Set to 0 to prevent RViz jitter (Static Frame)
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "velodyne"
        
        msg.height = 1
        msg.width = points.shape[0]
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name=field_name, offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        
        msg.is_dense = True
    
        # Optimized: Vectorized Struct Packing
        # Create a structured array directly that matches PointField layout
        # Fields: x, y, z, val (all float32)
        dtype_list = [
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            (field_name, np.float32)
        ]
        
        # Create empty array
        cloud_arr = np.empty(len(points), dtype=dtype_list)
        
        # Fill fields
        cloud_arr['x'] = points[:, 0]
        cloud_arr['y'] = points[:, 1]
        cloud_arr['z'] = points[:, 2]
        
        # Fill value
        # Ensure values match length
        if len(values) == len(points):
            cloud_arr[field_name] = values
        else:
            # Fallback or error?
            pass
            
        msg.data = cloud_arr.tobytes()
        return msg

    def publish_data(self):
        if hasattr(self, 'range_image'):
            # Convert to ROS Image
            # Float32 image
            img_msg = self.cv_bridge.cv2_to_imgmsg(self.range_image, encoding="32FC1")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "velodyne"
            
            self.range_image_pub.publish(img_msg)
            
        if hasattr(self, 'pc_msg'):
            self.pc_msg.header.stamp = self.get_clock().now().to_msg()
            self.point_cloud_pub.publish(self.pc_msg)
            
        if hasattr(self, 'wall_msg'):
            self.wall_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_walls.publish(self.wall_msg)
            
        if hasattr(self, 'void_msg') and self.void_msg:
            self.void_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_voids.publish(self.void_msg)
            
        if hasattr(self, 'delta_r_msg'):
            self.delta_r_msg.header.stamp = self.get_clock().now().to_msg()
            self.delta_r_cloud_pub.publish(self.delta_r_msg)
        
        if hasattr(self, 'delta_r_filtered_msg'):
            self.delta_r_filtered_msg.header.stamp = self.get_clock().now().to_msg()
            self.delta_r_filtered_pub.publish(self.delta_r_filtered_msg)
            
        if hasattr(self, 'ground_cloud_msg'):
            self.ground_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.ground_cloud_pub.publish(self.ground_cloud_msg)
        
        if hasattr(self, 'gt_cloud_msg'):
            self.gt_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.gt_cloud_pub.publish(self.gt_cloud_msg)
        
        if hasattr(self, 'bayes_cloud_msg'):
            self.bayes_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.bayes_cloud_pub.publish(self.bayes_cloud_msg)
        
        if hasattr(self, 'bayes_temporal_msg'):
            self.bayes_temporal_msg.header.stamp = self.get_clock().now().to_msg()
            self.bayes_temporal_pub.publish(self.bayes_temporal_msg)
            
        if hasattr(self, 'cluster_points_msg') and self.cluster_points_msg is not None:
            self.cluster_points_msg.header.stamp = self.get_clock().now().to_msg()
            self.cluster_points_pub.publish(self.cluster_points_msg)
        
        if hasattr(self, 'hull_marker_msg') and self.hull_marker_msg:
            self.hull_marker_msg.header.stamp = self.get_clock().now().to_msg()
            self.hull_pub.publish(self.hull_marker_msg)

        if hasattr(self, 'shadow_marker_msg') and self.shadow_marker_msg:
            self.shadow_marker_msg.header.stamp = self.get_clock().now().to_msg()
            self.shadow_pub.publish(self.shadow_marker_msg)

        if hasattr(self, 'shadow_cloud_msg') and self.shadow_cloud_msg:
            self.shadow_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.shadow_cloud_pub.publish(self.shadow_cloud_msg)
            
        if hasattr(self, 'text_marker_array_msg') and self.text_marker_array_msg:
            # for m in self.text_marker_array_msg.markers:
            #    m.header.stamp = self.get_clock().now().to_msg()
            # self.text_pub.publish(self.text_marker_array_msg) # DISABLED AS REQUESTED
            pass


    def probabilities_to_rgb(self, P_per_point, points):
        """Helper to convert probabilities to RGB colors."""
        r_b = np.full(len(points), 128, dtype=np.uint32)  # Default grey
        g_b = np.full(len(points), 128, dtype=np.uint32)
        b_b = np.full(len(points), 128, dtype=np.uint32)
        
        # Confirmed obstacle (P > 0.8)
        mask_conf_obs = P_per_point > 0.8
        r_b[mask_conf_obs] = 0; g_b[mask_conf_obs] = 0; b_b[mask_conf_obs] = 255
        
        # Probable obstacle (0.6 < P <= 0.8)
        mask_prob_obs = (P_per_point > 0.6) & (P_per_point <= 0.8)
        r_b[mask_prob_obs] = 100; g_b[mask_prob_obs] = 100; b_b[mask_prob_obs] = 255
        
        # Probably ground (0.2 <= P <= 0.4)
        mask_prob_gnd = (P_per_point >= 0.2) & (P_per_point <= 0.4)
        r_b[mask_prob_gnd] = 100; g_b[mask_prob_gnd] = 200; b_b[mask_prob_gnd] = 100
        
        # Confirmed ground (P < 0.2)
        mask_conf_gnd = P_per_point < 0.2
        r_b[mask_conf_gnd] = 0; g_b[mask_conf_gnd] = 200; b_b[mask_conf_gnd] = 0
        
        # Confirmed ground (P < 0.2)
        mask_conf_gnd = P_per_point < 0.2
        r_b[mask_conf_gnd] = 0; g_b[mask_conf_gnd] = 200; b_b[mask_conf_gnd] = 0
        
        rgb_bayes = (r_b << 16) | (g_b << 8) | b_b
        return rgb_bayes

    def cluster_objects(self, points, P_per_point, u_all, v_all):
        """
        Cluster confirmed obstacles (P > 0.6) and return a PointCloud2 
        where points are colored by their cluster ID. Ground clusters (< -1.5m) are filtered.
        """
        # 1. Filter points
        self.detected_clusters = [] # Reset clusters for this frame
        mask_obs = P_per_point > 0.6
        obs_points = points[mask_obs]
        
        # Extract indices for mapping back
        u_obs = u_all[mask_obs]
        v_obs = v_all[mask_obs]
        
        if len(obs_points) < 10:
            return None
            
        # 2. DBSCAN Clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(obs_points)
        labels = clustering.labels_
        
        # 3. Generate Colors & Filter
        r_c = np.full(len(obs_points), 255, dtype=np.uint32)
        g_c = np.full(len(obs_points), 255, dtype=np.uint32)
        b_c = np.full(len(obs_points), 255, dtype=np.uint32)
        
        unique_labels = set(labels)
        valid_mask = np.zeros(len(obs_points), dtype=bool)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask_lbl = labels == label
            pts = obs_points[mask_lbl]
            
            # Ground filter (center Z < -1.5)
            # Default ground is around -1.73
            if pts[:, 2].mean() < -1.5:
                continue
                
            # Valid cluster -> assign color and mark valid
            hue = (label * 77 % 360) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            r_c[mask_lbl] = int(r * 255)
            g_c[mask_lbl] = int(g * 255)
            b_c[mask_lbl] = int(b * 255)
            
            valid_mask[mask_lbl] = True
            
            # Store valid clusters for shadow generation
            self.detected_clusters.append(pts)
            
        if not np.any(valid_mask):
            return None
            
        final_points = obs_points[valid_mask]
        rgb_final = (r_c[valid_mask] << 16) | (g_c[valid_mask] << 8) | b_c[valid_mask]
        
        # --- SHADOW VERIFICATION ---
        # Update belief based on shadow scores
        self.update_belief_with_shadows(unique_labels, labels, obs_points, u_obs, v_obs, valid_mask)
        
        # Create Cloud
        cloud_msg = self.create_cloud(final_points, rgb_final.view(np.float32), field_name='rgb')
        return cloud_msg

    def update_belief_with_shadows(self, unique_labels, labels, obs_points, u_obs, v_obs, valid_mask):
        """
        Boosts belief map for clusters that cast strong shadows.
        """
        if self.concave_hull_indices is None:
            return

        # Use projected points to calculate angles
        x = obs_points[:, 0]
        y = obs_points[:, 1]
        theta = np.arctan2(y, x)
        r = np.linalg.norm(obs_points[:, :2], axis=1)
        
        text_markers = MarkerArray()
        m_id = 0
        
        for label in unique_labels:
            if label == -1: continue
            
            mask_c = labels == label
            
            # Check if valid (passed height filter)
            if not np.any(valid_mask[mask_c]):
                continue

            pts_c = obs_points[mask_c]
            theta_c = theta[mask_c]
            r_c = r[mask_c]
            
            # Calculate Shadow Score
            score = self.calculate_shadow_score(pts_c, theta_c, r_c)
            
            if score > 0.6: # Strong evidence
                # BOOST BELIEF (+2.0 log-odds)
                # Recover (u,v) indices for this cluster
                u_c = u_obs[mask_c]
                v_c = v_obs[mask_c]
                
                # Apply boost directly to belief map
                # (Clip to avoid overflow if needed, although update_belief clamps already)
                self.belief_map[u_c, v_c] += 2.0
                np.clip(self.belief_map, -10.0, 10.0, out=self.belief_map)
                
                # Create Text Marker
                tm = Marker()
                tm.header.frame_id = "velodyne"
                tm.ns = "shadow_scores"
                tm.id = m_id
                m_id += 1
                tm.type = Marker.TEXT_VIEW_FACING
                tm.action = Marker.ADD
                
                # Position above cluster
                cx = np.mean(pts_c[:, 0])
                cy = np.mean(pts_c[:, 1])
                cz = np.mean(pts_c[:, 2])
                
                tm.pose.position.x = float(cx)
                tm.pose.position.y = float(cy)
                tm.pose.position.z = float(cz + 1.0) # 1m above object
                tm.pose.orientation.w = 1.0
                tm.scale.z = 0.5 # Text height
                tm.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # White text
                tm.text = f"CONFIRMED\nScore: {score:.2f}"
                tm.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
                text_markers.markers.append(tm)

        if text_markers.markers:
             self.text_pub.publish(text_markers)
             self.text_marker_array_msg = text_markers

    def calculate_shadow_score(self, cluster_pts, theta_pts, r_pts):
        """
        Calculates fraction of empty/far pixels behind the cluster in the lower half of the scan.
        """
        if len(cluster_pts) < 5: return 0.0
        
        min_theta = np.min(theta_pts)
        max_theta = np.max(theta_pts)
        max_r_cluster = np.max(r_pts)
        
        # Sample angles across the cluster width
        steps = 10
        if max_theta - min_theta < 0.01: 
            steps = 3 # Small object
            
        angles = np.linspace(min_theta, max_theta, steps)
        
        shadow_pixels = 0
        total_pixels = 0
        
        for ang in angles:
            # Map angle to column index v
            # Formula: v = (1 - (theta + fov)/fov_tot) * W ... depends on sensor model.
            # Velodyne: theta goes from +PI to -PI usually in index 0..W
            # Approx mapping: v ~ 0.5 * (1.0 - ang / np.pi) * W
            v_col = int(0.5 * (1.0 - ang / np.pi) * self.W) % self.W
            
            # Get range values for this column
            column_ranges = self.current_range_image[:, v_col]
            
            # Look at LOWER HALF (u > H/2) where ground usually is
            lower_half_ranges = column_ranges[self.H//2:]
            
            # Check pixels that are geometrically "behind" the object
            # Condition: range > max_r_cluster + 1.0m (margin)
            # We count how many of these are EMPTY (inf) or blocked by hull logic.
            # Actually simpler: count proportion of 'inf' in lower half behind object.
            
            # Filter pixels that are behind the object
            behind_mask = lower_half_ranges > (max_r_cluster + 0.5)
            
            # Of the pixels behind, how many are empty (Infinite)?
            # Infinite means the laser didn't hit anything (or hit something very far > 120m).
            # If there was ground, it would return a finite value ~ r_ground.
            # So Infinite -> Shadow.
            
            pixels_behind = lower_half_ranges[behind_mask]
            if len(pixels_behind) == 0:
                continue
                
            n_inf = np.sum(np.isinf(pixels_behind))
            n_total = len(pixels_behind)
            
            shadow_pixels += n_inf
            total_pixels += n_total
            
        if total_pixels == 0: return 0.0
        
        return float(shadow_pixels) / float(total_pixels)

    # --- Concave Hull & Shadow Logic ---

    def compute_concave_hull(self, points):
        """
        Computes Concave Hull (Alpha Shape) using Optimized Frontier Sampling + Delaunay.
        Strategy:
        1. Downsample cloud to "Frontier Points" (Max Range per angular sector).
        2. Add Cluster Extreme Points to guarantee inclusion.
        3. Compute Delaunay on reduced set (~2000 points) -> Ultra Fast.
        4. Filter by Alpha and Extract Boundary.
        5. Smooth with Chaikin.
        """
        if len(points) < 4:
            return

        # 1. OPTIMIZATION: Frontier Sampling (Polar Downsampling)
        # Instead of feeding 100k points to Delaunay, we feed only the boundary candidates.
        # This reduces complexity from O(N log N) with N=100k to N=2k.
        
        n_sectors = 2048 # High resolution (0.17 deg) to preserve details
        
        xy = points[:, :2]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        
        # Filter global outliers by range (e.g. max 80m)
        valid_range_mask = r < 80.0
        points = points[valid_range_mask]
        xy = xy[valid_range_mask]
        r = r[valid_range_mask]
        theta = theta[valid_range_mask]
        
        if len(points) < 4: return
        
        # Binning
        theta_norm = (theta + np.pi) / (2 * np.pi)
        sector_idx = (theta_norm * n_sectors).astype(np.int32)
        sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
        
        # Select Max Range Point per Sector (Vectorized)
        # Sort by (sector, -r)
        sort_order = np.lexsort((-r, sector_idx))
        sorted_sectors = sector_idx[sort_order]
        sorted_indices = sort_order
        
        # Find unique sectors (first occurrence is max range due to sort)
        _, unique_indices = np.unique(sorted_sectors, return_index=True)
        
        frontier_indices = sorted_indices[unique_indices]
        # Map back to original points (which were already filtered by range, so indices match valid subset)
        frontier_points = points[frontier_indices, :2] # XY only
        
        # 2. Add Cluster Extremes (Anchors)
        # Essential to prevent clusters from forming disconnected islands if they are far features
        if hasattr(self, 'detected_clusters') and self.detected_clusters:
            cluster_extremes = []
            for c in self.detected_clusters:
                if len(c) < 1: continue
                
                c_xy = c[:, :2]
                min_x, min_y = np.min(c_xy, axis=0)
                max_x, max_y = np.max(c_xy, axis=0)
                # Add 4 corners of bbox + center to be safe
                cluster_extremes.append([min_x, min_y])
                cluster_extremes.append([max_x, max_y])
                cluster_extremes.append([min_x, max_y])
                cluster_extremes.append([max_x, min_y])
                # Also add the point closest to ego (min r) to help connect back if needed
                # and point furthest (max r)
                c_r = np.linalg.norm(c_xy, axis=1)
                cluster_extremes.append(c_xy[np.argmax(c_r)])
                cluster_extremes.append(c_xy[np.argmin(c_r)])
            
            if cluster_extremes:
                frontier_points = np.vstack([frontier_points, np.array(cluster_extremes)])

        # 3. Delaunay Triangulation (On reduced set)
        try:
            tri = Delaunay(frontier_points)
        except Exception as e:
            self.get_logger().error(f"Delaunay failed: {e}")
            return

        # 4. Filter Triangles by Alpha
        coords = frontier_points[tri.simplices]
        a = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
        b = np.linalg.norm(coords[:, 1] - coords[:, 2], axis=1)
        c = np.linalg.norm(coords[:, 2] - coords[:, 0], axis=1)
        
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        mask_valid = area > 1e-6
        
        r_circum = np.zeros_like(area)
        r_circum[mask_valid] = (a[mask_valid] * b[mask_valid] * c[mask_valid]) / (4.0 * area[mask_valid])
        r_circum[~mask_valid] = 9999.0
        
        # DYNAMIC ALPHA SHAPE (Adaptive Radius)
        # Instead of fixed radius, allow larger triangles at larger distances.
        # Density drops with distance^2, so gaps increase.
        
        # Calculate mean distance of each triangle from sensor
        # vertices: coords (M, 3, 2)
        dist_v1 = np.linalg.norm(coords[:, 0], axis=1)
        dist_v2 = np.linalg.norm(coords[:, 1], axis=1)
        dist_v3 = np.linalg.norm(coords[:, 2], axis=1)
        mean_dist = (dist_v1 + dist_v2 + dist_v3) / 3.0
        
        # Threshold: 20% of range (approx 12 deg gap coverage) + base 4.0m
        # At 10m: max_r = 4.0m (tight)
        # At 50m: max_r = 10.0m (relaxed)
        # At 80m: max_r = 16.0m (very relaxed)
        adaptive_threshold = np.maximum(4.0, mean_dist * 0.2)
        
        valid_triangles = r_circum < adaptive_threshold
        
        # 5. Extract Boundary
        simplices = tri.simplices[valid_triangles]
        if len(simplices) == 0: return
            
        edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
        edges.sort(axis=1)
        edges_view = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
        _, inverse, counts = np.unique(edges_view, return_inverse=True, return_counts=True)
        boundary_indices = np.where(counts == 1)[0]
        
        # Reconstruct unique boundary edges
        # We need the actual edge values corresponding to boundary_indices
        # Since 'counts' aligns with 'unique' values, we need to map back or re-find.
        # Faster way: Flatten edges, find unique with counts.
        
        unique_edges_view, unique_idx = np.unique(edges_view, return_index=True)
        boundary_edges_view = unique_edges_view[counts == 1]
        
        # Convert back to (N, 2) int array
        # This is strictly byte-data. We need to cast back to int.
        # But wait, edges_view was void. 'unique_edges_view' is also void.
        # We can just use the indices into the original 'edges' array found by 'unique_idx'
        # 'unique_idx' points to the *first* occurrence in 'edges' (or flattened unique).
        # Wait, np.unique(return_index=True) returns indices such that unique_arr = input[indices].
        
        # So:
        boundary_edge_indices_in_input = unique_idx[counts == 1]
        boundary_edges = edges[boundary_edge_indices_in_input]
        
        if len(boundary_edges) == 0: return
        
        # 6. Order Edges
        adj = {}
        for u, v in boundary_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            
        start_node = boundary_edges[np.argmin(frontier_points[boundary_edges[:, 0], 0]), 0]
        polygon_indices = [start_node]
        current = start_node
        visited = {start_node}
        
        while True:
            neighbors = adj.get(current, [])
            next_node = None
            for n in neighbors:
                if n == start_node and len(polygon_indices) > 2:
                    next_node = n; break
                if n not in visited:
                    next_node = n; break
            if next_node is None or next_node == start_node: break
            polygon_indices.append(next_node)
            visited.add(next_node)
            current = next_node
            
        polygon_indices = np.array(polygon_indices)
        
        # 7. Chaikin Smoothing
        poly_points = frontier_points[polygon_indices]
        smooth_poly = smooth_chaikin(poly_points, iterations=2, closed=True)
        
        # Store
        z_level = -1.0
        hull_points_3d = np.zeros((len(smooth_poly), 3))
        hull_points_3d[:, :2] = smooth_poly
        hull_points_3d[:, 2] = z_level
        
        self.current_points_3d = hull_points_3d
        self.points_2d = smooth_poly
        
        num_pts = len(smooth_poly)
        indices = np.arange(num_pts)
        next_indices = np.roll(indices, -1)
        self.concave_hull_indices = np.stack((indices, next_indices), axis=1)

        self.publish_hull_marker()

    def publish_hull_marker(self):
        if self.concave_hull_indices is None: return
        
        # Determine Frame ID
        frame_id = "velodyne"
        
        marker = self.create_visualization_marker(
            self.concave_hull_indices, 
            "concave_hull", 
            [1.0, 0.0, 1.0], # Magenta
            z_level=-1.7, # Ignored now
            frame_id=frame_id
        )
        
        if marker:
            self.hull_pub.publish(marker)
            self.hull_marker_msg = marker # Persist hull marker

    def create_visualization_marker(self, indices, ns, color_rgb, z_level=-0.1, frame_id="velodyne"):
        """
        Generates a ROS Marker for the hull (Replicated from hull_utils.py).
        """
        if indices is None: return None
        
        marker = Marker()
        marker.header.frame_id = frame_id
        # marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0
        marker.ns = ns
        marker.id = 0
        
        # Concave Hull = Set of edges -> LINE_LIST
        marker.type = Marker.LINE_LIST
             
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15 
        marker.color = ColorRGBA(r=float(color_rgb[0]), g=float(color_rgb[1]), b=float(color_rgb[2]), a=1.0)
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() # Infinite lifetime
        marker.frame_locked = False # Removed frame locking to simplify

        # Support Set, List, or Numpy Array
    # if isinstance(indices, (set, list, np.ndarray)): # cleaner check
    
        # Debug:
        # self.get_logger().info(f"Visualizing Indices: Type={type(indices)}, Len={len(indices) if indices is not None else 'None'}")
        
        # Extract edges
        edges_to_draw = indices
        if isinstance(indices, set):
            edges_to_draw = list(indices)
            
        # self.get_logger().info(f"Edges to draw: {len(edges_to_draw)}")
        
        for i, edge in enumerate(edges_to_draw):
            idx1 = int(edge[0])
            idx2 = int(edge[1])
            
            # Use stored 3D points for actual Z
            if hasattr(self, 'current_points_3d'):
                 p1 = self.current_points_3d[idx1]
                 p2 = self.current_points_3d[idx2]
                 z1 = float(p1[2])
                 z2 = float(p2[2])
            else:
                 # Fallback
                 z1 = float(z_level)
                 z2 = float(z_level)
            
            p1_2d = self.points_2d[idx1]
            p2_2d = self.points_2d[idx2]
            
            pt1 = Point(x=float(p1_2d[0]), y=float(p1_2d[1]), z=z1)
            pt2 = Point(x=float(p2_2d[0]), y=float(p2_2d[1]), z=z2)
            
            marker.points.append(pt1)
            marker.points.append(pt2)
        
        # self.get_logger().info(f"Marker points: {len(marker.points)}")
        return marker

    def compute_cluster_shadow(self, cluster_points, context_r=None, context_theta=None):
        """
        Generates shadow polygon vertices for a cluster (Ported from hull_utils.py).
        """
        if len(cluster_points) == 0: return []
        
        # 1. Polar Conversion
        r = np.linalg.norm(cluster_points[:, :2], axis=1)
        # Fix z-lookup for cluster shadows if needed, but here we use VIS_Z_LEVEL
        theta = np.arctan2(cluster_points[:, 1], cluster_points[:, 0])
        
        # 2. Angular Range
        min_theta_idx = np.argmin(theta)
        max_theta_idx = np.argmax(theta)
        
        theta_min = theta[min_theta_idx]
        theta_max = theta[max_theta_idx]
        
        is_crossing = (theta_max - theta_min) > np.pi
        
        step_deg = 0.5
        step_rad = np.radians(step_deg)
        
        # Generate sampling angles
        if is_crossing:
            theta_pos = theta.copy()
            theta_pos[theta_pos < 0] += 2*np.pi
            t_min_pos = np.min(theta_pos)
            t_max_pos = np.max(theta_pos)
            angles_pos = np.arange(t_min_pos, t_max_pos + 1e-6, step_rad)
            angles = angles_pos.copy()
            angles[angles > np.pi] -= 2*np.pi
        else:
            angles = np.arange(theta_min, theta_max + 1e-6, step_rad)
            
        # 3. Shape Fitting
        fan_points = []
        tolerance_rad = step_rad
        
        for ang in angles:
            # A. Start (d_near)
            if is_crossing:
                diff = np.abs(np.arctan2(np.sin(theta - ang), np.cos(theta - ang)))
            else:
                diff = np.abs(theta - ang)
            
            mask_beam = diff < tolerance_rad
            if np.any(mask_beam):
                d_near = np.max(r[mask_beam])
            else:
                d_near = np.max(r)
            
            # B. End (d_far) from Hull
            d_hull = self.compute_ray_intersection(ang)
            d_far = d_hull
            
            # C. Check Occlusion (Simplification: Skip complex context check for now or use full points)
            if context_r is not None and context_theta is not None:
                a1 = ang - tolerance_rad
                a2 = ang + tolerance_rad
                
                # Simple check within bounds
                if -np.pi <= a1 and a2 <= np.pi:
                    idx_start = np.searchsorted(context_theta, a1)
                    idx_end = np.searchsorted(context_theta, a2)
                    
                    if idx_end > idx_start:
                        cand_r = context_r[idx_start:idx_end]
                        mask_behind = cand_r > (d_near + 0.5)
                        if np.any(mask_behind):
                            d_obstacle = np.min(cand_r[mask_behind])
                            d_far = min(d_hull, d_obstacle)
            
            d_far = max(d_far, d_near + 0.1)
            
            p_in = Point(x=d_near*math.cos(ang), y=d_near*math.sin(ang), z=self.VIS_Z_LEVEL)
            p_out = Point(x=d_far*math.cos(ang), y=d_far*math.sin(ang), z=self.VIS_Z_LEVEL)
            fan_points.append((p_in, p_out))
            
        return fan_points

    def generate_geometric_shadows(self, all_points=None):
        """
        Generates shadow markers (Ported from hull_utils.py).
        """
        if not self.detected_clusters: return None
        
        # Pre-process context
        context_r = None
        context_theta = None
        
        if all_points is not None and len(all_points) > 0:
            cr = np.linalg.norm(all_points[:, :2], axis=1)
            ct = np.arctan2(all_points[:, 1], all_points[:, 0])
            sort_idx = np.argsort(ct)
            context_r = cr[sort_idx]
            context_theta = ct[sort_idx]
        
        marker = Marker()
        # Determine Frame ID
        frame_id = "velodyne"
        marker.header.frame_id = frame_id
        
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "geometric_shadows"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.5) # Azure semi-transparent
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        for cluster_points in self.detected_clusters:
            fan = self.compute_cluster_shadow(cluster_points, context_r, context_theta)
            
            for i in range(len(fan) - 1):
                p1_in, p1_out = fan[i]
                p2_in, p2_out = fan[i+1]
                
                # Triangle 1
                marker.points.append(p1_in)
                marker.points.append(p1_out)
                marker.points.append(p2_out)
                
                # Triangle 2
                marker.points.append(p1_in)
                marker.points.append(p2_out)
                marker.points.append(p2_in)
                
        return marker

    def detect_geometric_shadows(self, range_image, u_all, v_all, points, d_per_point):
        """
        Implements Geometric Shadow Logic with Slope Handling (Optimized).
        Uses pre-computed height maps to avoid inner-loop math.
        """
        shadow_boost_map = np.zeros((self.H, self.W), dtype=np.float32)
        shadow_points = []
        
        H, W = range_image.shape
        
        # --- VECTORIZED PRE-COMPUTATION ---
        
        # 1. Height Map Construction (H, W)
        # We need local_d for every pixel. Map d_per_point to image.
        # Initialize with default 1.73
        d_img = np.full((H, W), 1.73, dtype=np.float32)
        d_img[u_all, v_all] = d_per_point
        
        # Map Z to image
        z_img = np.full((H, W), -999.0, dtype=np.float32)
        z_img[u_all, v_all] = points[:, 2]
        
        # Calculate Height Map (z + d)
        height_map = z_img + d_img
        
        # 2. Geometric Maps (Pitch, Yaw)
        # Pitch depends on Row (u)
        u_indices = np.arange(H).reshape(-1, 1)
        pitch_map = np.radians((1.0 - u_indices / H) * 28.0 - 25.0) # (H, 1) broadcastable
        
        # Pre-compute Sine of Pitch for Shadow Projection logic
        sin_neg_pitch_map = np.sin(-pitch_map)
        
        # Yaw depends on Col (v) is not strictly needed for logic, only for projection?
        # Only needed for shadow END point calculation (r_gnd_expected).
        # r_gnd_expected = local_d / sin(-pitch). Yaw not needed for R.
        
        # 3. Obstacle Mask (Vectorized)
        mask_valid = (range_image > 0.5) & (range_image < 60.0)
        mask_obj = mask_valid & (height_map > 0.2)
        
        # --- 4. GROUND DROP-OUT DETECTION (Negative Obstacles) ---
        # Detects missing ground (holes, black ice, absorbing objects)
        # where model predicts ground but sensor sees nothing.
        
        # Avoid division by zero and horizon
        sin_p = sin_neg_pitch_map
        valid_proj = sin_p > 0.08 # >4.5 deg (only near ground)
        
        r_exp_map = np.full((H, W), 999.0, dtype=np.float32)
        # Safe division
        np.divide(d_img, sin_p, out=r_exp_map, where=valid_proj)
        
        # Condition:
        # 1. We expect ground nearby (< 15m)
        # 2. But sensor sees NOTHING (Invalid range)
        mask_void_obstacle = (~mask_valid) & valid_proj & (r_exp_map < 15.0)
        
        # Mark as Obstacle
        shadow_boost_map[mask_void_obstacle] += 0.8
        
        # Generate 3D points for Visualization of Voids
        if np.any(mask_void_obstacle):
            u_void, v_void = np.where(mask_void_obstacle) # u=row(pitch), v=col(yaw)
            r_void = r_exp_map[mask_void_obstacle]
            
            # Reconstruct 3D
            # pitch depends on u: pitch_map[u, 0]
            p_void = pitch_map[u_void, 0]
            # yaw depends on v: need to reconstruct logic from project_points_to_uv?
            # yaw = (v / W - 0.5) * 2 * pi ? No.
            # proj_x = 0.5 * (yaw / pi + 1.0) -> v = proj_x * W
            # v / W = 0.5 * (yaw/pi + 1)
            # 2 * v / W - 1 = yaw / pi
            # yaw = (2 * v / W - 1) * pi
            y_void = (2.0 * v_void / float(W) - 1.0) * np.pi
            
            # Spherical to Cartesian
            # x = r * cos(pitch) * cos(yaw)
            # y = r * cos(pitch) * sin(yaw)
            # z = r * sin(pitch)
            rcos = r_void * np.cos(p_void)
            x_v = rcos * np.cos(y_void)
            y_v = rcos * np.sin(y_void)
            z_v = r_void * np.sin(p_void)
            
            self.void_points = np.column_stack((x_v, y_v, z_v))
        else:
            self.void_points = []
            
        # --- COLUMN-WISE SCAN ---
        # We still loop over columns, but inner math is now lookups.
        
        
        # Iterate over columns (v) - 1024 iterations (Fast)
        for v in range(W):
            # Scan from Bottom (Near, u=H-1) to Top (Far, u=0)
            u_iter = range(H - 1, -1, -1)
            
            # State Machine: 0=Search, 1=Track, 2=Validate
            state = 0
            
            obs_r = -1.0
            obs_h_max = -999.0
            obs_u_top = -1
            shadow_end_r = 0.0
            
            # Local D for this column (approximate from valid pixels?)
            # We can pick d from the object base. 
            # Or use d_img[u, v].
            
            col_r = range_image[:, v]
            col_h = height_map[:, v]
            col_obj_mask = mask_obj[:, v]
            col_valid = mask_valid[:, v]
            col_d = d_img[:, v]
            col_sin_neg_p = sin_neg_pitch_map[:, 0] # (H,)
            
            for u in u_iter:
                # Fast Lookups
                r_val = col_r[u]
                is_obj = col_obj_mask[u]
                is_valid_px = col_valid[u]
                
                # --- State Logic ---
                
                if state == 0: # Searching
                    if is_obj:
                        # Found Object Base
                        state = 1
                        obs_r = r_val
                        obs_h_max = col_h[u]
                        obs_u_top = u
                        
                elif state == 1: # Tracking Object
                    if is_valid_px and abs(r_val - obs_r) < 1.0:
                        # Still on same object
                        obs_h_max = max(obs_h_max, col_h[u])
                        obs_u_top = u
                        obs_r = min(obs_r, r_val)
                    else:
                        # Object Ended -> Calculate Shadow
                        # Local D at object top/base
                        local_d_at_obj = col_d[obs_u_top]
                        
                        if obs_h_max >= (local_d_at_obj - 0.1):
                            L_shadow = 50.0 
                        else:
                            L_shadow = obs_r * obs_h_max / (local_d_at_obj - obs_h_max)
                        L_shadow = min(L_shadow, 50.0)
                        shadow_end_r = obs_r + L_shadow
                        
                        state = 2 # Start Validating current pixel (fallthrough conceptually)
                        # We process THIS pixel as first shadow candidate immediately?
                        # Yes, let's reuse logic below
                        
                if state == 2: # Shadow Validation
                    # Check if this pixel is inside shadow range
                    # r_gnd_expected
                    sin_p = col_sin_neg_p[u]
                    
                    if sin_p > 0:
                        r_gnd_expected = col_d[u] / sin_p
                    else:
                        r_gnd_expected = 999.9
                        
                    if r_gnd_expected < shadow_end_r:
                        # Inside Shadow Zone
                        # Check for Void (Shadow evidence) or Ground (Conflict)
                        
                        if not is_valid_px:
                            # Void -> Confirmed Shadow
                            # Apply Boost
                             dist_factor = max(self.shadow_min_decay, 1.0 - (obs_r / self.shadow_decay_dist))
                             shadow_boost_map[obs_u_top, v] += 0.5 * dist_factor
                        
                        elif abs(r_val - r_gnd_expected) < 0.5:
                            # Ground found -> No shadow -> Suppress
                            # Use low penalty allows tolerating thin fences with ground behind
                             shadow_boost_map[obs_u_top, v] -= 0.2
                    
                    else:
                        # End of shadow
                        state = 0 # Reset
                        
                        # Re-process this pixel as new candidate?
                        # Re-process this pixel as new candidate?
                        if is_obj:
                             state = 1
                             obs_r = r_val
                             obs_h_max = col_h[u]
                             obs_u_top = u
                             
        # Normalize/Scale map
        # Accumulation might be small (0.5 per pixel). 
        # But we only boost 'obs_u_top' pixel.
        # So 2.0 per CONFIRMED pixel is better if we want instant result.
        
        # Post-Scaling: if map > X -> prob -> 1.0
        return shadow_boost_map * 4.0 # Scale up to ensure effect
        
        # Clip boost map
        np.clip(shadow_boost_map, -5.0, 5.0, out=shadow_boost_map)
        
        # Publish Shadow Cloud
        if len(shadow_points) > 0:
            shadow_points_np = np.array(shadow_points, dtype=np.float32)
            rgb_blue = (0 << 16) | (0 << 8) | 255
            rgb_float = np.array([rgb_blue], dtype=np.uint32).view(np.float32)[0]
            rgbs = np.full(len(shadow_points), rgb_float, dtype=np.float32)
            
            self.shadow_cloud_msg = self.create_cloud(shadow_points_np, rgbs, field_name='rgb')
            self.shadow_cloud_pub.publish(self.shadow_cloud_msg)
        else:
             self.shadow_cloud_msg = None
             
        return shadow_boost_map

    def compute_ray_intersection(self, angle_rad):
        """
        Calculates intersection of a ray with the Concave Hull.
        """
        if self.concave_hull_indices is None or self.points_2d is None:
            return self.MAX_RANGE

        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        
        pts = self.points_2d
        
        # Optimized: Use array directly
        edge_indices = self.concave_hull_indices
        if len(edge_indices) == 0:
             return self.MAX_RANGE
        
        hp1 = pts[edge_indices[:,0]]
        hp2 = pts[edge_indices[:,1]]
        
        v = hp2 - hp1
        denom = v[:,0]*dy - v[:,1]*dx
        
        valid_denom = np.abs(denom) > 1e-6
        if not np.any(valid_denom):
            return self.MAX_RANGE
            
        p1x = hp1[valid_denom, 0]
        p1y = hp1[valid_denom, 1]
        vx = v[valid_denom, 0]
        vy = v[valid_denom, 1]
        curr_denom = denom[valid_denom]
        
        t = (p1y * vx - p1x * vy) / curr_denom
        u = (p1y * dx - p1x * dy) / curr_denom
        
        mask_inter = (t > 0) & (u >= 0.0) & (u <= 1.0)
        
        if np.any(mask_inter):
            valid_t = t[mask_inter]
            return np.min(valid_t)
            
        return self.MAX_RANGE

    def save_evaluation_metrics(self):
        """
        Guarda métricas de detección para evaluación offline.

        Exporta:
        - belief_prob: (H, W) probabilidad final del Bayes Filter
        - points: (N, 3) nube de puntos del último frame
        - obstacle_mask: (N,) máscara booleana de obstáculos detectados
        """
        if not hasattr(self, 'belief_prob') or self.belief_prob is None:
            self.get_logger().warn("[save_evaluation_metrics] belief_prob no disponible")
            return

        # Directorio de salida
        output_dir = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/tests/range_projection_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guardar belief_prob (H, W)
        belief_file = output_dir / f"belief_prob_scan_{self.current_scan}.npy"
        np.save(belief_file, self.belief_prob)
        self.get_logger().info(f"[✓] Guardado belief_prob: {belief_file}")

        # Guardar points (N, 3)
        if hasattr(self, 'points') and self.points is not None:
            points_file = output_dir / f"points_scan_{self.current_scan}.npy"
            np.save(points_file, self.points)
            self.get_logger().info(f"[✓] Guardado points: {points_file}")

        # Crear obstacle_mask per-point desde belief_map
        # Threshold: P > 0.5 → obstacle
        threshold_prob = 0.5
        belief_mask_2d = self.belief_prob > threshold_prob  # (H, W)

        # Proyectar a per-point mask
        if hasattr(self, 'points') and hasattr(self, 'u') and hasattr(self, 'v'):
            N = len(self.points)
            obstacle_mask = np.zeros(N, dtype=bool)

            for i in range(N):
                ui = self.u[i]
                vi = self.v[i]
                if 0 <= ui < self.belief_prob.shape[0] and 0 <= vi < self.belief_prob.shape[1]:
                    obstacle_mask[i] = belief_mask_2d[ui, vi]

            mask_file = output_dir / f"obstacle_mask_scan_{self.current_scan}.npy"
            np.save(mask_file, obstacle_mask)
            self.get_logger().info(f"[✓] Guardado obstacle_mask: {mask_file}")
            self.get_logger().info(f"    Obstacles detectados: {obstacle_mask.sum()} / {N} ({100*obstacle_mask.sum()/N:.1f}%)")
        else:
            self.get_logger().warn("[save_evaluation_metrics] No se pudo crear obstacle_mask (falta self.u o self.v)")

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Range View Projection with Bayesian Temporal Filter')
    parser.add_argument('--data_path', default='/home/insia/lidar_ws/data_odometry_velodyne/dataset', help='Path to SemanticKITTI dataset root')
    parser.add_argument('--scene', default='00', help='Sequence Number')
    parser.add_argument('--scan_start', default='0', help='First scan index')
    parser.add_argument('--scan_end', default='4', help='Last scan index (inclusive, default=4 -> 5 frames)')
    # Keep --scan for backwards compatibility (single frame mode)
    parser.add_argument('--scan', default=None, help='Single scan index (overrides scan_start/scan_end)')
    
    # Filter out ROS args
    clean_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    parsed_args = parser.parse_args(clean_args)
    
    # Single-scan backwards compatibility
    if parsed_args.scan is not None:
        scan_start = int(parsed_args.scan)
        scan_end = int(parsed_args.scan)
    else:
        scan_start = int(parsed_args.scan_start)
        scan_end = int(parsed_args.scan_end)
    
    node = RangeViewNode(
        data_path=parsed_args.data_path,
        scene=parsed_args.scene,
        scan_start=scan_start,
        scan_end=scan_end
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
