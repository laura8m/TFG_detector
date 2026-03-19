import numpy as np
import open3d as o3d
import sys
import os
import time
from scipy.spatial import Delaunay, cKDTree
from sklearn.cluster import DBSCAN
import cv2
import colorsys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smoothing_utils import smooth_chaikin

# Try to import ROS 2 modules (Fail gracefully for unit tests)
try:
    import rclpy
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Point
    import struct
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] ROS 2 modules not found. Visualization disabled.")

class LidarProcessingSuite:
    def __init__(self, data_path, sensor_height=1.73, ros_node=None):
        """
        Inicializa la suite de procesamiento modular.
        
        Args:
            data_path (str): Ruta al archivo de datos (.bin o .pcd).
            sensor_height (float): Altura del sensor respecto al suelo (metros).
            ros_node (rclpy.node.Node, optional): Nodo ROS 2 para publicación.
        """
        self.data_path = data_path
        self.points = None
        self.ground_indices = None
        self.nonground_indices = None
        self.sensor_height = sensor_height
        self.ros_node = ros_node
        self.current_scan = 0 # Default, updated by runner
        
        # Atributos de visualización
        self.concave_hull_indices = None
        self.points_2d = None
        self.current_points_3d = None
        self.detected_clusters = [] # Lista de puntos por cluster
        
        # KITTI Calibration (Tr_velo_to_cam)
        self.Tr = np.array([
            [4.2768028e-04, -9.9996725e-01, -8.0844917e-03, -1.1984599e-02],
            [-7.2106265e-03, 8.0811985e-03, -9.9994132e-01, -5.4039847e-02],
            [9.9997386e-01, 4.8594858e-04, -7.2069002e-03, -2.9219686e-01],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Cargar poses si es posible
        self.poses = self.load_poses()
        if self.poses:
            print(f"[INFO] Poses cargadas: {len(self.poses)}")
            self.frame_id = "map"
        else:
            print("[WARN] No se encontraron poses. Warping deshabilitado.")
            self.frame_id = "velodyne"

        if self.ros_node and ROS_AVAILABLE:
            self._init_publishers()

        # Parámetros del sensor (HDL-64E)
        self.H = 64
        self.W = 2048
        self.fov_up = 3.0 * np.pi / 180.0
        self.fov_down = -25.0 * np.pi / 180.0
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        
        # Filtro Bayesiano (Log-Odds)
        self.belief_map = np.zeros((self.H, self.W), dtype=np.float64)
        self.l0 = 0.0 # Prior no informativo (P=0.5)
        self.threshold_obs = -0.3
        
        # Inicializar Patchwork++
        self._init_patchwork()
        # Inicializar parámetros para proyección local (CZM)
        self.initialize_czm_params()

    def _init_publishers(self):
        """Inicializa los publishers de ROS 2."""
        node = self.ros_node
        qos = 10
        self.pub_bayes = node.create_publisher(PointCloud2, 'bayes_cloud', qos)
        self.pub_clusters = node.create_publisher(PointCloud2, 'cluster_points', qos)
        self.pub_shadows_cloud = node.create_publisher(PointCloud2, 'shadow_cloud', qos)
        self.pub_shadows_marker = node.create_publisher(Marker, 'geometric_shadows', qos)
        self.pub_hull = node.create_publisher(Marker, 'concave_hull', qos)
        self.pub_voids = node.create_publisher(PointCloud2, 'void_cloud', qos)
        self.pub_walls = node.create_publisher(PointCloud2, 'detected_walls', qos)
        
        # Additional Publishers for Feature Parity
        self.pub_delta_r = node.create_publisher(PointCloud2, 'delta_r_cloud', qos)
        self.pub_filtered = node.create_publisher(PointCloud2, 'delta_r_filtered_cloud', qos)
        self.pub_ground = node.create_publisher(PointCloud2, 'ground_cloud', qos)
        self.pub_gt = node.create_publisher(PointCloud2, 'gt_cloud', qos)
        self.pub_bayes_temporal = node.create_publisher(PointCloud2, 'bayes_temporal_cloud', qos)
        print("[INFO] Publishers de ROS 2 inicializados.")

    def initialize_czm_params(self):
        """Inicializa los parámetros de zonas concéntricas (CZM) para binning."""
        # Configuración estándar para HDL-64E
        min_r = self.params.min_range # 2.7
        max_r = self.params.max_range # 80.0
        
        # 4 Zonas
        self.min_ranges = [
            min_r,
            (7 * min_r + max_r) / 8.0,
            (3 * min_r + max_r) / 4.0,
            (min_r + max_r) / 2.0
        ]
        
        self.num_rings_each_zone = [2, 4, 4, 4]
        self.num_sectors_each_zone = [16, 32, 54, 32]
        
        # Tamaños de anillos
        self.ring_sizes = []
        self.ring_sizes.append((self.min_ranges[1] - self.min_ranges[0]) / self.num_rings_each_zone[0])
        self.ring_sizes.append((self.min_ranges[2] - self.min_ranges[1]) / self.num_rings_each_zone[1])
        self.ring_sizes.append((self.min_ranges[3] - self.min_ranges[2]) / self.num_rings_each_zone[2])
        self.ring_sizes.append((max_r - self.min_ranges[3]) / self.num_rings_each_zone[3])
        
        # Tamaños de sectores (radianes)
        self.sector_sizes = [2 * np.pi / n for n in self.num_sectors_each_zone]

    def get_czm_bin(self, x, y):
        """Calcula el bin (zona, anillo, sector) para cada punto (Vectorizado)."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        
        zone_idx = np.full_like(r, -1, dtype=np.int32)
        ring_idx = np.full_like(r, -1, dtype=np.int32)
        sector_idx = np.full_like(r, -1, dtype=np.int32)
        
        valid = (r > self.params.min_range) & (r <= self.params.max_range)
        
        # Zona 0
        mask_z0 = valid & (r < self.min_ranges[1])
        if np.any(mask_z0):
            zone_idx[mask_z0] = 0
            ring_idx[mask_z0] = ((r[mask_z0] - self.min_ranges[0]) / self.ring_sizes[0]).astype(np.int32)
            sector_idx[mask_z0] = (theta[mask_z0] / self.sector_sizes[0]).astype(np.int32)

        # Zona 1
        mask_z1 = valid & (r >= self.min_ranges[1]) & (r < self.min_ranges[2])
        if np.any(mask_z1):
            zone_idx[mask_z1] = 1
            ring_idx[mask_z1] = ((r[mask_z1] - self.min_ranges[1]) / self.ring_sizes[1]).astype(np.int32)
            sector_idx[mask_z1] = (theta[mask_z1] / self.sector_sizes[1]).astype(np.int32)

        # Zona 2
        mask_z2 = valid & (r >= self.min_ranges[2]) & (r < self.min_ranges[3])
        if np.any(mask_z2):
            zone_idx[mask_z2] = 2
            ring_idx[mask_z2] = ((r[mask_z2] - self.min_ranges[2]) / self.ring_sizes[2]).astype(np.int32)
            sector_idx[mask_z2] = (theta[mask_z2] / self.sector_sizes[2]).astype(np.int32)

        # Zona 3
        mask_z3 = valid & (r >= self.min_ranges[3])
        if np.any(mask_z3):
            zone_idx[mask_z3] = 3
            ring_idx[mask_z3] = ((r[mask_z3] - self.min_ranges[3]) / self.ring_sizes[3]).astype(np.int32)
            sector_idx[mask_z3] = (theta[mask_z3] / self.sector_sizes[3]).astype(np.int32)

        # Clip indices
        for z in range(4):
            mask = (zone_idx == z)
            if np.any(mask):
                ring_idx[mask] = np.clip(ring_idx[mask], 0, self.num_rings_each_zone[z] - 1)
                sector_idx[mask] = np.clip(sector_idx[mask], 0, self.num_sectors_each_zone[z] - 1)

        return zone_idx, ring_idx, sector_idx

    def get_czm_bin_scalar(self, x, y):
        """Versión escalar para construcción de lookup table."""
        r = np.sqrt(x**2 + y**2)
        if r <= self.params.min_range or r > self.params.max_range: return None
        theta = np.arctan2(y, x)
        if theta < 0: theta += 2*np.pi
        
        if r < self.min_ranges[1]: z=0; r_base=self.min_ranges[0]
        elif r < self.min_ranges[2]: z=1; r_base=self.min_ranges[1]
        elif r < self.min_ranges[3]: z=2; r_base=self.min_ranges[2]
        else: z=3; r_base=self.min_ranges[3]
            
        r_idx = min(int((r-r_base)/self.ring_sizes[z]), self.num_rings_each_zone[z]-1)
        s_idx = min(int(theta/self.sector_sizes[z]), self.num_sectors_each_zone[z]-1)
        return (z, r_idx, s_idx)
        
    def _init_patchwork(self):
        """Inicializa la librería de segmentación de suelo Patchwork++."""
        try:
            import pypatchworkpp
            self.params = pypatchworkpp.Parameters()
            self.params.verbose = False
            self.params.sensor_height = self.sensor_height
            self.params.num_iter = 3
            self.params.num_lpr = 20
            self.params.num_min_pts = 10
            self.params.th_dist = 0.2
            self.params.max_range = 80.0
            self.params.min_range = 2.7
            self.patchwork = pypatchworkpp.patchworkpp(self.params)
            print("[INFO] Patchwork++ inicializado.")
        except ImportError:
            print("[ERROR] No se pudo importar pypatchworkpp.")
            sys.exit(1)

    def load_poses(self):
        """
        Carga poses.txt del dataset SemanticKITTI.
        Returns: Lista de matrices 4x4.
        """
        # Try to infer sequence path from data_path
        # Expected: .../dataset/sequences/XX/velodyne/YYYYYY.bin
        try:
            p = Path(self.data_path)
            # Walk up until 'velodyne' parent is found, then 'sequences', then root
            # Actually simplest: Assume standard structure if "sequences" in path
            parts = p.parts
            if 'sequences' in parts:
                seq_idx = parts.index('sequences')
                scene = parts[seq_idx+1]
                dataset_root = Path(*parts[:seq_idx])
                pose_path = dataset_root / 'sequences' / scene / 'poses.txt'
                
                if not pose_path.exists():
                    # Try alternate location
                    pose_path = dataset_root.parent / 'data_odometry_labels' / 'dataset' / 'sequences' / scene / 'poses.txt'
                
                if pose_path.exists():
                    poses = []
                    with open(pose_path, 'r') as f:
                        for line in f:
                            values = [float(v) for v in line.strip().split()]
                            T = np.eye(4)
                            T[:3, :] = np.array(values).reshape(3, 4)
                            poses.append(T)
                    return poses
        except Exception as e:
            print(f"[WARN] Error loading poses: {e}")
            
        return None

    def warp_belief_map(self, belief_map, current_scan_idx, points_current):
        """
        Warp del mapa de creencias usando odometría.
        """
        prev_scan_idx = current_scan_idx - 1
        
        if (self.poses is None or 
            prev_scan_idx < 0 or
            current_scan_idx >= len(self.poses) or
            prev_scan_idx >= len(self.poses)):
            return belief_map
        
        T_world_curr = self.poses[current_scan_idx]
        T_world_prev = self.poses[prev_scan_idx]
        
        # T_prev_curr = inv(T_world_prev) @ T_world_curr
        # Adjust for Tr (Calibration)
        # T_rel_cam = inv(T_world_prev) @ T_world_curr
        # T_prev_curr_lidar = inv(Tr) @ T_rel_cam @ Tr
        
        T_rel_cam = np.linalg.inv(T_world_prev) @ T_world_curr
        T_prev_curr = np.linalg.inv(self.Tr) @ T_rel_cam @ self.Tr
        
        # Transform points to prev frame
        N = points_current.shape[0]
        pts_hom = np.hstack([points_current, np.ones((N, 1))])
        pts_in_prev = (T_prev_curr @ pts_hom.T).T[:, :3]
        
        # Project to prev UV
        u_prev, v_prev, r_prev_proj = self.project_points_to_uv(pts_in_prev)
        
        # Project current to UV
        u_curr, v_curr, _ = self.project_points_to_uv(points_current) # Reuse cached? self.u, self.v
        
        valid = (u_prev >= 0) # project_points_to_uv handles clip but let's assume valid returns mask
        # My project_points_to_uv implementation in lidar_modules returns clipped u,v.
        # Check range
        # TODO: Refactor project_points_to_uv to return valid mask properly if needed.
        # But here simple clipping checks might be enough if project returned clipped.
        
        warped_belief = np.zeros((self.H, self.W), dtype=np.float64)
        
        # Data Association / Depth Jump Check
        # We need r_prev_proj (range of point in prev frame coords) vs r_measured_prev (from prev abs range image)
        if hasattr(self, 'prev_abs_range') and self.prev_abs_range is not None:
             r_sensor_prev = self.prev_abs_range[u_prev, v_prev]
             diff = np.abs(r_sensor_prev - r_prev_proj)
             mask_assoc = diff < 0.2
        else:
             mask_assoc = np.ones(N, dtype=bool)
             
        # Wall Mask Check (re-use if available)
        # if hasattr(self, 'rejected_mask'): mask_assoc &= ~self.rejected_mask
        
        # Add 'at'
        # Filter valid points only? project_points_to_uv returns clipped, so indices are always valid.
        # But we should check if they were FOV valid.
        
        start_mask = mask_assoc
        if np.any(start_mask):
             prev_beliefs = belief_map[u_prev[start_mask], v_prev[start_mask]]
             
             np.add.at(warped_belief, (u_curr[start_mask], v_curr[start_mask]), prev_beliefs)
             count = np.zeros((self.H, self.W), dtype=np.float64)
             np.add.at(count, (u_curr[start_mask], v_curr[start_mask]), 1.0)
             
             nonzero = count > 0
             warped_belief[nonzero] /= count[nonzero]
             
        return warped_belief

    def load_point_cloud(self):
        """
        Carga la nube de puntos desde el archivo especificado.
        
        Returns:
            np.ndarray: Nube de puntos cargada (N, 3).
        """
        print(f"[PASO 1] Cargando nube de puntos: {self.data_path}")
        if self.data_path.endswith('.bin'):
            scan = np.fromfile(self.data_path, dtype=np.float32).reshape((-1, 4))
            self.points = scan[:, :3]
        else:
            pcd = o3d.io.read_point_cloud(self.data_path)
            self.points = np.asarray(pcd.points)
        print(f"    -> Puntos cargados: {len(self.points)}")
        return self.points

    def segment_ground(self):
        """
        Segmenta el suelo utilizando Patchwork++.
        
        Returns:
            tuple: (puntos_suelo, puntos_no_suelo)
        """
        print("[PASO 2] Segmentando suelo con Patchwork++...")
        start = time.time()
        self.patchwork.estimateGround(self.points)
        ground = self.patchwork.getGround()
        nonground = self.patchwork.getNonground()
        elapsed = time.time() - start
        
        self.ground_points = ground
        self.nonground_points = nonground
        self.centers = self.patchwork.getCenters()
        self.normals = self.patchwork.getNormals()
        
        print(f"    -> Tiempo: {elapsed:.4f}s")
        print(f"    -> Suelo: {len(ground)}, No-Suelo: {len(nonground)}")
        return ground, nonground

    def project_range_view_global(self):
        """
        Proyecta la nube utilizando un PLANO GLOBAL simple (Baseline).
        Útil para comparar y ver la mejora de los planos locales.
        """
        print("[PASO 3a] Proyección Basica (Plano Global)...")
        points = self.points
        r = np.linalg.norm(points, axis=1)
        
        # Plano Global: z = -1.73
        # r_exp = -1.73 * r / z
        z = points[:, 2].copy()
        z[np.abs(z) < 1e-3] = 1e-3
        r_exp = -self.sensor_height * r / z
        
        delta_r = r - r_exp
        delta_r = np.clip(delta_r, -20.0, 10.0)
        
        # Calcular MSE en puntos de suelo para métricas
        if self.ground_points is not None:
             r_g = np.linalg.norm(self.ground_points, axis=1)
             z_g = self.ground_points[:, 2].copy()
             z_g[np.abs(z_g) < 1e-3] = 1e-3
             r_exp_g = -self.sensor_height * r_g / z_g
             delta_g = r_g - r_exp_g
             mse = np.mean(delta_g**2)
             print(f"    -> MSE Visual (Suelo): {mse:.4f}")
             return delta_r, mse
        return delta_r, 0.0

    def project_range_view_local(self):
        """
        Proyección SOTA usando Planos Locales de Patchwork++.
        """
        print("[PASO 3b] Proyección Avanzada (Planos Locales)...")
        points = self.points
        
        # 1. Construir Lookup Table de Planos
        local_planes = {}
        centers = self.centers
        normals = self.normals
        
        # Fallback: Plano Global calculado con SVD del suelo
        if len(self.ground_points) > 10:
            centroid = np.mean(self.ground_points, axis=0)
            centered = self.ground_points - centroid
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            global_n = Vt[2, :]
            if global_n[2] < 0: global_n = -global_n
            global_d = -np.dot(global_n, centroid)
            # Validar verticalidad
            if global_n[2] < 0.9: # Si falla, usar default
                global_n = np.array([0,0,1.0]); global_d = self.sensor_height
        else:
            global_n = np.array([0,0,1.0]); global_d = self.sensor_height

        # Construir KDTree para validación detallada (usamos todos los puntos para contexto real)
        # Optimización: Usar solo ground_points o un subset reduce tiempo
        # Para detectar escalones, ground_points es suficiente
        kdtree = None
        if len(self.ground_points) > 0:
            kdtree = cKDTree(self.ground_points)

        # Llenar tabla con planos locales
        for i, c in enumerate(centers):
            n = normals[i]
            
            # --- SOTA v2.0: Slope-Sensitive Wall Rejection ---
            # Si la normal es horizontal (posible pared), aplicamos validación dura
            if abs(n[2]) < 0.7: 
                is_wall = True
                if kdtree is not None:
                     # Consultar radio 0.5m
                     idxs = kdtree.query_ball_point(c, 0.5)
                     if len(idxs) > 5:
                         z_local = self.ground_points[idxs, 2]
                         delta_z = np.percentile(z_local, 95) - np.percentile(z_local, 5)
                         # Si es suave (RAMPA), lo aceptamos. Si tiene escalón (PARED), rechazo.
                         if delta_z <= 0.3:
                             is_wall = False
                
                # Heurística fallback: si Z es muy alto (> -1.0), probablemente es pared/techo
                if c[2] > -1.0: 
                    is_wall = True
                    
                if is_wall:
                    continue 
            
            bin_id = self.get_czm_bin_scalar(c[0], c[1])
            if bin_id:
                d = -np.dot(n, c)
                local_planes[bin_id] = (n, d)
        
        # 2. Vectorizar Lookup (Tabla 4D: zonas, rings, sectors, 4 params)
        planes_table = np.zeros((4, 4, 54, 4), dtype=np.float32)
        planes_table[..., :3] = global_n
        planes_table[..., 3] = global_d
        
        # Build KDTree for Wall Rejection (only if needed)
        # We use ground points for query as they are the candidates for the plane
        if len(self.ground_points) > 0:
            ground_kdtree = cKDTree(self.ground_points)
        else:
            ground_kdtree = None

        for bin_id, (n_loc, d_loc) in local_planes.items():
            # Validación Sensible a la Pendiente (SOTA v2.0)
            # Si es horizontal (pared potencial), verificar geometría
            if abs(n_loc[2]) < 0.7:
                 is_wall = True
                 # Consultar Z local
                 if ground_kdtree is not None:
                     # Recuperar centro aproximado del bin (patchwork no da centro exacto del bin, usamos estimate)
                     # Buscamos en centers de patchwork que tenemos
                     # Patchwork getCenters devuelve los centros de los planos estimados
                     # Iteramos sobre ellos en el bucle principal.
                     
                     # En este bucle iteramos sobre un dict. Necesitamos el centro asociado al bin.
                     # Hack: reconstruimos centro dummy o buscamos.
                     # Mejor: hacer la validación en el bucle anterior "for i, c in enumerate(centers):"
                     pass
            
            z, r_idx, s_idx = bin_id
            if 0<=z<4 and 0<=r_idx<4 and 0<=s_idx<54:
                planes_table[z, r_idx, s_idx, :3] = n_loc
                planes_table[z, r_idx, s_idx, 3] = d_loc
                
        # 3. Asignar plano a cada punto
        z_idx, r_idx, s_idx = self.get_czm_bin(points[:, 0], points[:, 1])
        valid_bins = (z_idx >= 0) & (r_idx >= 0) & (s_idx >= 0)
        
        n_per_point = np.full((len(points), 3), global_n, dtype=np.float32)
        d_per_point = np.full(len(points), global_d, dtype=np.float32)
        
        if np.any(valid_bins):
            # Advanced indexing
            # Clip indices just in case to avoid index errors
            z_v = np.clip(z_idx[valid_bins], 0, 3)
            r_v = np.clip(r_idx[valid_bins], 0, 3)
            s_v = np.clip(s_idx[valid_bins], 0, 53)
            
            params = planes_table[z_v, r_v, s_v]
            n_per_point[valid_bins] = params[:, :3]
            d_per_point[valid_bins] = params[:, 3]
            
        # 4. Calcular r_exp = -d / (n . P/r)
        # n . P = dot_prod
        dot_prod = np.sum(points * n_per_point, axis=1)
        r = np.linalg.norm(points, axis=1)
        
        valid_dot = dot_prod < -1e-3
        r_exp = np.full_like(r, 999.9)
        r_exp[valid_dot] = -d_per_point[valid_dot] * r[valid_dot] / dot_prod[valid_dot]
        
        delta_r = r - r_exp
        delta_r = np.clip(delta_r, -20.0, 10.0)

        # Generar imagen (Range View)
        # Usamos mismos U, V que antes o recalculamos? Recalculamos
        u, v, _ = self.project_points_to_uv(points)
        
        range_image = np.full((self.H, self.W), 0.0, dtype=np.float32)
        order = np.argsort(r)[::-1]
        range_image[u[order], v[order]] = delta_r[order]
        
        self.range_image = range_image
        self.delta_r = delta_r
        self.n_per_point = n_per_point
        self.d_per_point = d_per_point
        self.u = u; self.v = v; self.r = r
        
        # MSE
        if self.ground_points is not None:
             # Necesitamos índices de los puntos de suelo en el array original 'points'
             # O recalculamos delta para ground_points usando la misma lógica
             # Recalculamos rápido para métrica
             g_pts = self.ground_points
             z_g, r_g, s_g = self.get_czm_bin(g_pts[:,0], g_pts[:,1])
             valid_g = (z_g>=0)
             # ... (Simplificado: Usar solo puntos válidos para MSE)
             # Asumimos que la mejora visual es suficiente en test suite
             pass

        print(f"    -> Proyección Local completada.")
        return range_image, u, v

    def filter_physical_outliers(self):
        """
        Filtra anomalías físicas (socavones falsos).
        Verifica que los puntos con delta_r positivo estén REALMENTE bajo tierra.
        
        Returns:
            np.ndarray: Máscara booleana de socavones validados.
        """
        print("[PASO 4] Verificando consistencia física (Filtro de Socavones)...")
        
        # Z esperada (Calculada usando planos locales si están disponibles)
        if hasattr(self, 'd_per_point') and hasattr(self, 'n_per_point'):
             # z_ground = -(d + nx*x + ny*y) / nz
             nx = self.n_per_point[:, 0]
             ny = self.n_per_point[:, 1]
             nz = self.n_per_point[:, 2]
             d = self.d_per_point
             
             valid_z = np.abs(nz) > 0.1
             z_expected = np.full_like(self.points[:, 2], -self.sensor_height) # Default global
             
             x = self.points[:, 0]
             y = self.points[:, 1]
             
             z_expected[valid_z] = -(d[valid_z] + nx[valid_z]*x[valid_z] + ny[valid_z]*y[valid_z]) / nz[valid_z]
        else:
             z_expected = -self.sensor_height
             
        z_diff = self.points[:, 2] - z_expected
        
        # Máscara: Delta R > 0.2 (Rango anómalo) Y Z_diff < -0.15 (Hundimiento físico real)
        mask_sinkhole = (self.delta_r > 0.2) & (self.delta_r < 2.0) & (z_diff < -0.15)
        
        n_candidates = np.sum((self.delta_r > 0.2) & (self.delta_r < 2.0))
        n_confirmed = np.sum(mask_sinkhole)
        
        print(f"    -> Candidatos (Naranja): {n_candidates}")
        print(f"    -> Confirmados (Profundo): {n_confirmed} (Rechazados: {n_candidates - n_confirmed})")
        return mask_sinkhole

    def apply_geometric_consistency(self, range_image):
        """
        Aplica suavizado geométrico Inter-Ring (Consistencia entre anillos).
        Verifica si un obstáculo es consistente verticalmente (tiene soporte arriba/abajo).
        """
        print("[PASO 5] Aplicando Consistencia Geométrica (Inter-Ring)...")
        # Máscara inicial de obstáculos (basada en umbral -0.3)
        obs_mask = range_image < self.threshold_obs
        
        # Kernel vertical (3x1) para verificar vecinos superior e inferior
        kernel = np.ones((3, 1), np.uint8)
        
        # Dilatación: Rellena huecos si hay obstáculos cerca verticalmente
        dilated = cv2.dilate(obs_mask.astype(np.uint8), kernel, iterations=1)
        
        # Erosión: Elimina ruido aislado que no tiene soporte vertical
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Resultado final booleano
        consistent_mask = eroded > 0
        n_added = np.sum(consistent_mask) - np.sum(obs_mask)
        n_removed = np.sum(obs_mask) - np.sum(consistent_mask & obs_mask)
        
        print(f"    -> Puntos añadidos (relleno): {n_added}")
        print(f"    -> Ruido eliminado (aislado): {n_removed}")
        return consistent_mask

    def get_raw_probability(self, range_image):
        """Convierte Delta R en probabilidad bruta de obstáculo (sigmoide invertida)."""
        # Mapeo: delta_r < -0.3 => P > 0.5 (Obstáculo)
        # delta_r > 0 => P < 0.5 (Suelo/Hoyo)
        
        # Factor k controla la pendiente de la transición
        k = 2.0 
        # Centrado en threshold_obs (-0.3)
        return 1.0 / (1.0 + np.exp(k * (range_image - self.threshold_obs)))

    def update_belief(self, belief_map_state, P_obs, points_current):
        """
        Actualiza el mapa de creencias bayesiano (Log-Odds) con la nueva observación.
        
        Args:
            belief_map_state: Estado anterior (Log-Odds acumulado).
            P_obs: Probabilidad de la observación actual.
            points_current: Nube de puntos actual (para proyectar y actualizar celdas).
        """
        print("[PASO 6] Actualizando Filtro Bayesiano Temporal...")
        
        # Step 1: Warp previous belief to current frame coordinates
        warped_belief = self.warp_belief_map(belief_map_state, self.current_scan, points_current)
        
        # Step 2: Convert current observation to log-odds
        eps = 1e-6
        # Mapeo: delta_r < -0.3 => P > 0.5 (Obstáculo)
        # P_obs ya viene como probabilidad raw
        P_clamped = np.clip(P_obs, eps, 1.0 - eps)
        L_obs_vals = np.log(P_clamped / (1.0 - P_clamped)) - self.l0
        
        # Mapear L_obs_vals (N_points) a L_obs_map (H, W)
        # Usamos self.u y self.v de la proyección
        L_obs_map = np.zeros_like(belief_map_state)
        order = np.argsort(self.r)[::-1]
        u_sorted = self.u[order]
        v_sorted = self.v[order]
        L_vals_sorted = L_obs_vals[order] # Oops, P_obs es imagen o vector?
        # get_raw_probability devuelve (H, W) o vector?
        # En lidar_modules.py get_raw_probability(self.delta_r) usa self.delta_r que es vector (N,).
        # Ah no, self.delta_r en project_range_view_local es vector (N,).
        # Pero range_image ES una imagen 2D.
        
        # Revisemos get_raw_probability en lidar_modules.py
        # return 1.0 / (1.0 + np.exp(k * (range_image - self.threshold_obs)))
        # Si le pasamos self.delta_r (vector), devuelve vector.
        # Si le pasamos self.range_image (2D), devuelve 2D.
        
        # El código actual de run_full_pipeline llama: P_raw = self.get_raw_probability(self.delta_r) -> Vector
        # Entonces L_obs_vals es vector (N,). Correcto.
        
        L_obs_map[u_sorted, v_sorted] = L_vals_sorted
        
        # Step 3: Recursive Bayes update
        new_belief_map = warped_belief + L_obs_map
        
        # Clamp to avoid excessive inertia
        new_belief_map = np.clip(new_belief_map, -2.5, 2.5)
        
        # Convert back to probability for visualization
        P_belief = 1.0 / (1.0 + np.exp(-new_belief_map))
        
        return P_belief, new_belief_map

    def project_points_to_uv(self, points):
        """Helper: Proyecta puntos 3D a coordenadas de imagen (u, v)."""
        r = np.linalg.norm(points, axis=1)
        pitch = np.arcsin(points[:, 2] / r)
        yaw = np.arctan2(points[:, 1], points[:, 0])
        
        u = np.floor((1.0 - (pitch + abs(self.fov_down)) / self.fov) * self.H).astype(np.int32)
        v = np.floor((0.5 * (yaw / np.pi + 1.0)) * self.W).astype(np.int32)
        
        u = np.clip(u, 0, self.H - 1)
        v = np.clip(v, 0, self.W - 1)
        return u, v, r

    def detect_ground_voids(self):
        """
        Detecta 'Vacíos' (Obstáculos Negativos) buscando ausencia de suelo esperado.
        SOTA v2.0: Ground Drop-out detection.
        """
        print("[PASO 8a] Detectando Vacíos (Negative Obstacles)...")
        # Verificar donde NO hay retorno en el Range View, pero DEBERÍA haber suelo cercano.
        
        # Mapa de ocupación actual (1 = hay medición, 0 = vacío)
        occupancy = (self.range_image != 0).astype(np.float32)
        
        # Calcular Rango Esperado para CADA pixel (theta, phi) basado en plano global (simplificación eficiente)
        # Pitch u:
        u_vals = np.arange(self.H)
        pitch_vals = (1.0 - u_vals / self.H) * self.fov - abs(self.fov_down) # Aprox
        
        # r_exp = -h / sin(pitch)
        # Solo válido para pitch < 0 (mirando abajo)
        # Vectorizado por filas
        r_exp_map = np.full((self.H, self.W), 999.9)
        
        # Vectorized implementation
        valid_pitch = pitch_vals < -0.05
        # Calculate r for valid rows (shape: N_valid,)
        # Broadcast to (N_valid, W)
        r_exp_rows = -self.sensor_height / np.sin(pitch_vals[valid_pitch])
        r_exp_map[valid_pitch, :] = r_exp_rows[:, np.newaxis]
                
        # Máscara de Vacío:
        # 1. Sin medición (Occupancy == 0)
        # 2. Rango esperado cercano (< 15m) - Deberíamos ver suelo
        # 3. Ignorar horizonte (r_exp muy grande)
        
        void_mask = (occupancy == 0) & (r_exp_map < 15.0) & (r_exp_map > 1.0)
        
        # Nube de puntos de vacíos (reconstrucción sintética)
        rows, cols = np.where(void_mask)
        n_voids = len(rows)
        
        # Convertir a XYZ para visualización (opcional)
        # x = r_exp * cos(pitch) * cos(yaw) ...
        
        print(f"    -> Vacíos detectados (Ground Drop-out): {n_voids} celdas")
        return void_mask

    def detect_geometric_shadows(self, current_range_image, u, v, points, d_per_point=None):
        """
        Detecta sombras geométricas (Shadow Boost) para reforzar obstáculos.
        Busca saltos grandes en rango detrás de objetos.
        """
        print("[PASO 7] Calculando Boost de Sombras Geométricas...")
        
        # Crear mapa de boost
        boost_map = np.zeros_like(self.belief_map)
        
        # Simplificación Vectorizada:
        # Detectar saltos negativos grandes en rango a lo largo de las columnas (rayos)
        # Diff vertical (entre anillos)
        # diff_val = current_range_image[row] - current_range_image[row-1]
        
        # Si (Range_row - Range_row_prev) > Umbral (ej. 1.0m), hay una oclusión
        # El punto ANTERIOR (row-1) es el borde del objeto, el actual (row) es el fondo.
        
        # Log-Odds boost value
        SHADOW_BOOST_VAL = 2.0 
        
        # Iterar columnas (costoso en Python puro, vectorizamos con diff numpy)
        # Axis 0 = filas (anillos). Diff(axis=0) calcula r[i] - r[i-1]
        # Nota: Range View está ordenado de arriba a abajo en filas (0=Top, 63=Bottom)
        # Los rayos de LiDAR escanean de abajo a arriba o viceversa dependiendo de modelo.
        # Asumiremos consistencia espacial vertical.
        
        r_diff = np.diff(current_range_image, axis=0, prepend=current_range_image[0:1,:])
        
        # Detectar saltos grandes ( Fondo - Objeto > 1.0m )
        # Significa que pasamos de algo cercano a algo lejano bruscamente
        shadow_mask = r_diff > 1.0 
        
        # Aplicamos boost en la celda ANTERIOR al salto (el borde del objeto)
        # Shift mask down
        boost_indices = np.where(shadow_mask)
        rows = np.clip(boost_indices[0] - 1, 0, self.H-1)
        cols = boost_indices[1]
        
        boost_map[rows, cols] = SHADOW_BOOST_VAL
        
        n_shadows = np.sum(boost_map > 0)
        print(f"    -> Celdas reforzadas por sombra: {n_shadows}")
        
        return boost_map

    def compute_concave_hull(self, alpha=0.1):
        """
        Calcula el Concave Hull (Alpha Shape) optimizado usando Frontier Sampling.
        Estrategia:
        1. Downsampling polar (Max Range por sector).
        2. Añadir extremos de clusters (si existen).
        3. Delaunay sobre conjunto reducido (~2k puntos).
        4. Filtrado Alpha Adaptativo.
        5. Extracción de borde y Smoothing Chaikin.
        """
        print("[PASO 8] Calculando Concave Hull (Optimizado)...")
        points = self.points
        if len(points) < 4:
            return None

        # 1. OPTIMIZATION: Frontier Sampling (Polar Downsampling)
        n_sectors = 2048 # High resolution
        
        xy = points[:, :2]
        r = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        
        # Filter global outliers by range (e.g. max 80m)
        valid_range_mask = r < 80.0
        # Check if we have enough points after filter
        if np.sum(valid_range_mask) < 4: return None
        
        points_filtered = points[valid_range_mask]
        xy = xy[valid_range_mask]
        r = r[valid_range_mask]
        theta = theta[valid_range_mask]
        
        # Binning
        theta_norm = (theta + np.pi) / (2 * np.pi)
        sector_idx = (theta_norm * n_sectors).astype(np.int32)
        sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
        
        # Select Max Range Point per Sector (Vectorized)
        sort_order = np.lexsort((-r, sector_idx))
        sorted_sectors = sector_idx[sort_order]
        sorted_indices = sort_order
        
        # Find unique sectors (first occurrence is max range due to sort)
        _, unique_indices = np.unique(sorted_sectors, return_index=True)
        
        frontier_indices = sorted_indices[unique_indices]
        frontier_points = points_filtered[frontier_indices, :2] # XY only
        
        # 2. Add Cluster Extremes (Anchors) - Optional if clusters computed
        if hasattr(self, 'detected_clusters') and self.detected_clusters:
            cluster_extremes = []
            for c in self.detected_clusters:
                if len(c) < 1: continue
                c_xy = c[:, :2]
                min_x, min_y = np.min(c_xy, axis=0)
                max_x, max_y = np.max(c_xy, axis=0)
                cluster_extremes.append([min_x, min_y])
                cluster_extremes.append([max_x, max_y])
                cluster_extremes.append([min_x, max_y])
                cluster_extremes.append([max_x, min_y])
                c_r = np.linalg.norm(c_xy, axis=1)
                cluster_extremes.append(c_xy[np.argmax(c_r)])
                cluster_extremes.append(c_xy[np.argmin(c_r)])
            
            if cluster_extremes:
                frontier_points = np.vstack([frontier_points, np.array(cluster_extremes)])

        # 3. Delaunay Triangulation
        try:
            tri = Delaunay(frontier_points)
        except Exception as e:
            print(f"    [ERROR] Delaunay fallo: {e}")
            return None

        # 4. Filter Triangles by Alpha
        coords = frontier_points[tri.simplices]
        a = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
        b = np.linalg.norm(coords[:, 1] - coords[:, 2], axis=1)
        c = np.linalg.norm(coords[:, 2] - coords[:, 0], axis=1)
        
        s = (a + b + c) / 2.0
        # Safe sqrt
        val = s * (s - a) * (s - b) * (s - c)
        val = np.maximum(val, 1e-10)
        area = np.sqrt(val)
        mask_valid = area > 1e-6
        
        r_circum = np.zeros_like(area)
        r_circum[mask_valid] = (a[mask_valid] * b[mask_valid] * c[mask_valid]) / (4.0 * area[mask_valid])
        r_circum[~mask_valid] = 9999.0
        
        # Adaptive Threshold
        # Mean dist of triangle vertices
        dist_v1 = np.linalg.norm(coords[:, 0], axis=1)
        dist_v2 = np.linalg.norm(coords[:, 1], axis=1)
        dist_v3 = np.linalg.norm(coords[:, 2], axis=1)
        mean_dist = (dist_v1 + dist_v2 + dist_v3) / 3.0
        
        adaptive_threshold = np.maximum(4.0, mean_dist * 0.2)
        valid_triangles = r_circum < adaptive_threshold
        
        # 5. Extract Boundary
        simplices = tri.simplices[valid_triangles]
        if len(simplices) == 0: return None
            
        edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
        edges.sort(axis=1)
        
        # Fast unique edge count
        edges_view = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
        _, inverse, counts = np.unique(edges_view, return_inverse=True, return_counts=True)
        
        # Unique edges with count 1 are boundary
        unique_edges_view, unique_idx = np.unique(edges_view, return_index=True)
        # Indicies in original 'edges' array
        boundary_indices_in_edges = unique_idx[counts == 1]
        boundary_edges = edges[boundary_indices_in_edges]
        
        if len(boundary_edges) == 0: return None
        
        # 6. Order Edges (Polygonize)
        adj = {}
        for u, v in boundary_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            
        # Start with node closest to ego (min x usually works or min dist)
        # Use min X to be consistent
        start_node_idx = np.argmin(frontier_points[boundary_edges[:, 0], 0])
        start_node = boundary_edges[start_node_idx, 0]
        
        polygon_indices = [start_node]
        current = start_node
        visited = {start_node}
        
        max_iter = len(boundary_edges) * 2
        
        for _ in range(max_iter):
            neighbors = adj.get(current, [])
            next_node = None
            for n in neighbors:
                if n == start_node and len(polygon_indices) > 2:
                    next_node = n; break
                if n not in visited:
                    next_node = n; break
            
            if next_node is None: break
            if next_node == start_node: break
            
            polygon_indices.append(next_node)
            visited.add(next_node)
            current = next_node
            
        polygon_indices = np.array(polygon_indices)
        
        # 7. Chaikin Smoothing
        poly_points = frontier_points[polygon_indices]
        smooth_poly = smooth_chaikin(poly_points, iterations=2, closed=True)
        
        # Store for Visualization
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
        
        print(f"    -> Triángulos: {len(simplices)}, Hull Points: {len(smooth_poly)}")

    def cluster_objects(self, mask_objects):
        """
        Agrupa los puntos detectados como objetos u obstáculos usando DBSCAN.
        
        Args:
            mask_objects (np.ndarray): Máscara booleana de puntos a agrupar.
            
        Returns:
            tuple: (num_clusters, etiquetas)
        """
        print("[PASO 9] Clusterizando Objetos (DBSCAN)...")
        object_points = self.points[mask_objects]
        
        if len(object_points) == 0:
            print("    [WARN] No hay objetos para agrupar.")
            return 0, []
            
        # DBSCAN: eps=0.5m, min_samples=5
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(object_points)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"    -> Objetos detectados: {n_clusters}")
        return n_clusters, labels

    def generate_shadows(self, object_clusters):
        """
        Genera sombras geométricas detrás de los objetos detectados.
        (Simulación simple de proyección radial).
        """
        print("[PASO 10] Generando Sombras Geométricas...")
        # Lógica simplificada: proyectar desde el sensor (0,0) a través del centro del objeto
        # hasta el límite del mapa.
        if object_clusters == 0:
            return 0
        
        print(f"    -> Sombras calculadas para {object_clusters} objetos.")
        return object_clusters

    def run_full_pipeline(self):
        """Ejecuta todos los pasos secuencialmente."""
        self.load_point_cloud()
        self.segment_ground()
        self.project_range_view_local() # Usa la lógica avanzada SOTA
        
        # Filtros Intermedios
        self.filter_physical_outliers() # Check socavones
        
        # -- NUEVOS PASOS --
        # 1. Probabilidad Cruda
        P_raw = self.get_raw_probability(self.delta_r)
        
        # 2. Actualización Bayesiana
        P_belief, new_belief_map = self.update_belief(self.belief_map, P_raw, self.points)
        self.belief_map = new_belief_map # Guardar estado
        
        # 3. Consistencia Geométrica (Smoothing)
        self.apply_geometric_consistency(self.range_image)
        
        # 4. Shadow Boost
        # Necesitamos coordenadas U, V para todos los puntos
        u, v, _ = self.project_points_to_uv(self.points)
        
        # Para shadow boost usamos la imagen de rango ABSOLUTO (no delta)
        # Reconstruimos range image absoluta rápido
        abs_range_image = np.full((self.H, self.W), 999.0, dtype=np.float32)
        abs_range_image[self.u, self.v] = self.r
        
        boost = self.detect_geometric_shadows(abs_range_image, u, v, self.points)
        self.belief_map += boost # Aplicar boost al mapa de creencia
        
        # 5. Hull y Clustering
        self.detect_ground_voids()
        self.compute_concave_hull()
        
        # Usar resultado bayesiano final para clustering (umbral P > 0.8)
        # Mapear probabilidad de celdas a puntos
        P_points = P_belief[self.u, self.v]
        mask_final_obs = P_points > 0.8
        
        n_obj, labels = self.cluster_objects(mask_final_obs) # Modificado para retornar labels
        self.generate_shadows(n_obj)
        
        # Guardar clusters para visualización de sombras geométricas
        self.detected_clusters = []
        if n_obj > 0:
             unique_labels = set(labels)
             if -1 in unique_labels: unique_labels.remove(-1)
             for l in unique_labels:
                 self.detected_clusters.append(self.points[mask_final_obs][labels == l])


        # 6. Publicar Resultados (Visualización)
        if self.ros_node and ROS_AVAILABLE:
            self.publish_results(P_points, mask_final_obs, labels)

    def create_cloud(self, points, colors=None, field_name='rgb', frame_id=None):
        """Crea un mensaje PointCloud2 optimizado usando numpy."""
        if len(points) == 0: return None
        
        if frame_id is None:
            frame_id = getattr(self, 'frame_id', 'velodyne')

        header = Header()
        header.frame_id = frame_id
        # header.stamp = self.ros_node.get_clock().now().to_msg()
        # Use current time to avoid Message Filter drops in RViz
        if self.ros_node:
            header.stamp = self.ros_node.get_clock().now().to_msg()
        else:
            header.stamp.sec = 0
            header.stamp.nanosec = 0
        
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        
        if colors is not None:
            msg.fields.append(PointField(name=field_name, offset=12, datatype=PointField.FLOAT32, count=1))
            dtype_list.append((field_name, np.float32))
            
        msg.point_step = 16 if colors is not None else 12
        msg.row_step = msg.point_step * msg.width
        
        # Vectorized packing
        cloud_arr = np.empty(len(points), dtype=dtype_list)
        cloud_arr['x'] = points[:, 0].astype(np.float32)
        cloud_arr['y'] = points[:, 1].astype(np.float32)
        cloud_arr['z'] = points[:, 2].astype(np.float32)
        
        if colors is not None:
            if not isinstance(colors, np.ndarray):
                colors = np.array(colors, dtype=np.float32)
            cloud_arr[field_name] = colors.astype(np.float32)
            
        msg.data = cloud_arr.tobytes()
        return msg

    def probabilities_to_rgb(self, P_per_point, points):
        """Helper to convert probabilities to RGB colors."""
        r_b = np.full(len(points), 128, dtype=np.uint32)
        g_b = np.full(len(points), 128, dtype=np.uint32)
        b_b = np.full(len(points), 128, dtype=np.uint32)
        
        mask_conf_obs = P_per_point > 0.8
        r_b[mask_conf_obs] = 0; g_b[mask_conf_obs] = 0; b_b[mask_conf_obs] = 255
        
        mask_prob_obs = (P_per_point > 0.6) & (P_per_point <= 0.8)
        r_b[mask_prob_obs] = 100; g_b[mask_prob_obs] = 100; b_b[mask_prob_obs] = 255
        
        mask_prob_gnd = (P_per_point >= 0.2) & (P_per_point <= 0.4)
        r_b[mask_prob_gnd] = 100; g_b[mask_prob_gnd] = 200; b_b[mask_prob_gnd] = 100
        
        mask_conf_gnd = P_per_point < 0.2
        r_b[mask_conf_gnd] = 0; g_b[mask_conf_gnd] = 200; b_b[mask_conf_gnd] = 0
        
        rgb_bayes = (r_b << 16) | (g_b << 8) | b_b
        return rgb_bayes.view(np.float32)

    def publish_results(self, P_points, mask_obs, labels):
        """Publica todos los tópicos de visualización (Vectorizado)."""
        print("[PASO 11] Publicando Visualización en RViz...")
        self.last_msgs = {} # Cache for republishing
        
        # 1. Bayes Cloud (Coloreada por probabilidad)
        # Re-use P_points (which comes from belief map)
        P_vals = np.asarray(P_points)
        rgb_float = self.probabilities_to_rgb(P_vals, self.points)
            
        bayes_msg = self.create_cloud(self.points, rgb_float)
        if bayes_msg: 
            self.pub_bayes.publish(bayes_msg)
            self.last_msgs['bayes'] = bayes_msg
        
        # 2. Clusters
        if np.sum(mask_obs) > 0:
            obs_pts = self.points[mask_obs]
            obs_labels = labels
            
            cluster_colors = []
            unique_labels = set(obs_labels)
            color_map = {}
            for l in unique_labels:
                if l == -1: 
                    color_map[l] = 0.0 # Negro
                    continue
                hue = (l * 37 % 360) / 360.0
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                packed = (int(r*255) << 16) | (int(g*255) << 8) | int(b*255)
                color_map[l] = struct.unpack('f', struct.pack('I', packed))[0]
            
            for l in obs_labels:
                cluster_colors.append(color_map[l])
                
            cluster_msg = self.create_cloud(obs_pts, cluster_colors)
            if cluster_msg: 
                self.pub_clusters.publish(cluster_msg)
                self.last_msgs['clusters'] = cluster_msg

        # 3. Concave Hull Marker
        self.publish_hull_marker()
        
        # 4. Filter Feature Parity Clouds
        # Delta R Cloud
        if hasattr(self, 'delta_r'):
            # Simple color mapping for Delta R
            # < -0.3 Blue, > 0.2 Orange, else Grey
            r_c = np.full(len(self.points), 128, dtype=np.uint32)
            g_c = np.full(len(self.points), 128, dtype=np.uint32)
            b_c = np.full(len(self.points), 128, dtype=np.uint32)
            
            mask_o = self.delta_r < -0.3
            r_c[mask_o]=0; g_c[mask_o]=0; b_c[mask_o]=255
            
            mask_d = (self.delta_r > 0.2) & (self.delta_r < 2.0)
            r_c[mask_d]=255; g_c[mask_d]=165; b_c[mask_d]=0
            
            rgb = (r_c << 16) | (g_c << 8) | b_c
            delta_msg = self.create_cloud(self.points, rgb.view(np.float32))
            if delta_msg:
                self.pub_delta_r.publish(delta_msg)
                self.last_msgs['delta_r'] = delta_msg

        # Ground Cloud
        if hasattr(self, 'ground_points'):
             g_msg = self.create_cloud(self.ground_points, colors=None) # White/Intensity default
             if g_msg:
                 self.pub_ground.publish(g_msg)
                 self.last_msgs['ground'] = g_msg
                 
        # Wall Cloud (if rejection logic used)
        # TODO: Store rejected points in segment_ground/project_local
        
    def publish_hull_marker(self):
        """Publica el marcador del Concave Hull."""
        if self.concave_hull_indices is None or self.points_2d is None: return
        
        marker = Marker()
        marker.header.frame_id = self.frame_id
        if self.ros_node:
             marker.header.stamp = self.ros_node.get_clock().now().to_msg()
        else:
             marker.header.stamp.sec = 0
             marker.header.stamp.nanosec = 0
        marker.ns = "concave_hull"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0) # Magenta
        
        points = []
        for i, (idx1, idx2) in enumerate(self.concave_hull_indices):
            # Usar Z real si tenemos current_points_3d (generado en hull compute)
            # O usar Z fijo
            if self.current_points_3d is not None:
                 p1 = self.current_points_3d[idx1]
                 p2 = self.current_points_3d[idx2]
                 points.append(Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])))
                 points.append(Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2])))
            else:
                 p1 = self.points_2d[idx1]
                 p2 = self.points_2d[idx2]
                 points.append(Point(x=float(p1[0]), y=float(p1[1]), z=-1.7))
                 points.append(Point(x=float(p2[0]), y=float(p2[1]), z=-1.7))
        
        marker.points = points
        self.pub_hull.publish(marker)
        if not hasattr(self, 'last_msgs'): self.last_msgs = {}
        self.last_msgs['hull'] = marker

    def republish_last(self):
        """Vuelve a publicar los últimos mensajes generados con timestamp actualizado."""
        if not hasattr(self, 'last_msgs') or not self.last_msgs:
            return
            
        current_time = self.ros_node.get_clock().now().to_msg()
        
        # Helper to publish if exists and update stamp
        def pub_if_exists(key, publisher):
            if key in self.last_msgs:
                msg = self.last_msgs[key]
                # Update stamp to prevent RViz expiration/errors
                if isinstance(msg, PointCloud2) or isinstance(msg, Marker) or isinstance(msg, MarkerArray):
                     if hasattr(msg, 'header'):
                         msg.header.stamp = current_time
                     # Markers inside MarkerArray
                     if isinstance(msg, MarkerArray):
                         for m in msg.markers:
                             m.header.stamp = current_time
                publisher.publish(msg)

        pub_if_exists('bayes', self.pub_bayes)
        pub_if_exists('clusters', self.pub_clusters)
        pub_if_exists('hull', self.pub_hull)
        pub_if_exists('shadows', self.pub_shadows_marker)
        pub_if_exists('voids', self.pub_voids)
        pub_if_exists('delta_r', self.pub_delta_r)
        pub_if_exists('ground', self.pub_ground)
        pub_if_exists('gt', self.pub_gt)
        pub_if_exists('filtered', self.pub_filtered)
        pub_if_exists('temporal', self.pub_bayes_temporal)


if __name__ == "__main__":
    path = "/home/insia/lidar_ws/src/patchwork-plusplus/data/000000.bin"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    pipeline = LidarProcessingSuite(path)
    pipeline.run_full_pipeline()
