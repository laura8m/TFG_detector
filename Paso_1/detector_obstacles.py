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
from geometry_msgs.msg import Point, TransformStamped
import colorsys
from scipy.spatial import Delaunay, cKDTree
import math
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time as _time
import os
from tf2_ros import StaticTransformBroadcaster

# Añadir directorio actual al path para encontrar modulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smoothing_utils import smooth_chaikin

# Añadir Patchwork++ al path

class ObjectsDetector(Node):
    """
    Nodo principal para la detección de objetos.
    Realiza segmentación de suelo, proyección a imagen de rango, filtrado Bayesiano,
    detección de sombras geométricas y clustering.
    """
    def __init__(self, data_path, scene, scan_start, scan_end):
        super().__init__('objects_detector')
        
        # --- PARÁMETROS ---
        self.data_path = Path(data_path)
        self.scene = scene
        self.scan_start = int(scan_start)
        self.scan_end = int(scan_end)
        
        # Parámetros del Sensor (Velodyne HDL-64E)
        self.H = 64 # Altura de imagen (número de anillos del sensor)
        self.W = 2048 # Anchura de imagen (resolución horizontal)
        self.fov_up = 3.0 * np.pi / 180.0 # Campo de visión superior (+3 grados)
        self.fov_down = -25.0 * np.pi / 180.0 # Campo de visión inferior (-25 grados)
        self.fov = abs(self.fov_down) + abs(self.fov_up) # Rango vertical total (~28 grados)
        self.min_range = 2.7 # Distancia mínima (filtrar ego-vehículo)
        self.max_range = 80.0 # Distancia máxima efectiva
        
        # Parámetros de Segmentación de Suelo
        # -------------------------------------------------------------------------
        # Uso en el código:
        # 1. ground_z_threshold: Se usa en `process_clusters` para filtrar ruido.
        #    Si el centroide Z de un cluster es menor que este valor, se descarta.
        # 2. default_ground_height: Se pasa a Patchwork++ en `init_patchwork` como
        #    estimación inicial de la altura del sensor.
        # 3. wall_rejection_slope: Se usa en `analyze_local_planes`. Si la normal
        #    del plano tiene una componente Z baja (muy vertical, < 0.7), es una PARED.
        #    IMPORTANTE: Patchwork++ a veces ajusta planos a paredes o muros si son dominantes.
        #    Este parámetro evita clasificar muros verticales como "suelo transitable".
        # 4. wall_height_diff_threshold: Se usa en `analyze_local_planes` para
        #    verificar la rugosidad/varianza en Z de los puntos del plano.
        #    Evita que arbustos densos o rocas sean clasificados como planos de suelo. 
        #    Mide la variación real de altura (Z_max - Z_min) de los puntos asignados a ese plano. 
        #    Si los puntos suben y bajan más de 30cm (0.3m), asumimos que es una superficie rugosa/no transitable y la marcamos como obstáculo.
        # -------------------------------------------------------------------------
        self.ground_z_threshold = -1.5 # Umbral simple de altura Z para filtrado grosero (puntos por debajo de esto son candidatos a suelo)
        self.default_ground_height = 1.73 # Altura estimada del sensor sobre el suelo (usado como semilla para Patchwork++)
        self.wall_rejection_slope = 0.7 # Umbral de pendiente para rechazar paredes. Si el componente Z de la normal es < 0.7 (~45 grados), se considera muy inclinado para ser suelo.
        self.wall_height_diff_threshold = 0.3 # Umbral de rugosidad local. Si la diferencia de altura (Z_max - Z_min) en un parche local > 0.3m, se considera obstáculo/pared.
        
        # Parámetros del Filtro Bayesiano
        # -------------------------------------------------------------------------
        # Uso en el código:
        # 1. l0: Valor base de log-odds. `log(0.5/0.5) = 0`. Representa incertidumbre total inicial.
        # 2. threshold_obs: Umbral para Delta R. Si (R_medido - R_esperado) < -0.3m,
        #    el punto está significativamente más cerca de lo que debería ser si fuera suelo -> Obstáculo.
        # 3. belief_clamp_min/max: Satura el valor de log-odds para que nunca sea infinito.
        #    Permite que el mapa de creencia se actualice si la situación cambia (ej. un objeto se mueve).
        # 4. prob_threshold_obs/gnd: Umbrales finales para convertir la probabilidad continua (0.0-1.0)
        #    en decisiones binarias (Obstáculo vs Suelo) para clustering y visualización.
        # -------------------------------------------------------------------------
        self.l0 = 0.0 # Log-odds a priori
        self.threshold_obs = -0.3 # Umbral Delta R
        self.belief_clamp_min = -5.0
        self.belief_clamp_max = 5.0
        self.prob_threshold_obs = 0.6 # Probabilidad de obstáculo confirmado
        self.prob_threshold_gnd = 0.4 # Probabilidad de suelo confirmado
        
        # Parámetros de Control Temporal (Odometría)
        # -------------------------------------------------------------------------
        # Uso en el código:
        # 1. depth_jump_threshold: Umbral para el "Depth Jump Check" en `warp_belief_map`.
        #    Si al proyectar la memoria del pasado, la profundidad difiere más de 0.2m 
        #    con la medición actual, se descarta la memoria (evita ghosting).
        # -------------------------------------------------------------------------
        self.depth_jump_threshold = 0.2 
        
        # Parámetros de Sombras
        # -------------------------------------------------------------------------
        # Uso en el código:
        # 1. shadow_decay_dist: Distancia máxima de confianza.
        #    Se usa en `scan_column_for_shadows`. Si un obstáculo está muy lejos (>60m),
        #    su sombra es menos fiable porque el láser es menos preciso.
        #    La confianza decae linealmente hasta esta distancia.
        # 2. shadow_min_decay: Factor mínimo de confianza.
        #    Incluso a la distancia máxima, la sombra aporta al menos un 20% de su valor base.
        # 3. shadow_score_threshold: Umbral de validación para `check_cluster_shadow`.
        #    Un cluster debe tener al menos el 60% de sus píxeles traseros como "vacío" o "infinito"
        #    para confirmar que es un obstáculo sólido que bloquea el láser.
        # 4. shadow_boost_val: "Premio" de probabilidad.
        #    Si se confirma una sombra geométrica clara, sumamos +2.0 (en log-odds) al mapa de creencia
        #    en la posición del objeto, aumentando drásticamente la certeza de que existe.
        # -------------------------------------------------------------------------
        self.shadow_decay_dist = 60.0 # Distancia de decaimiento
        self.shadow_min_decay = 0.2
        self.shadow_score_threshold = 0.6
        self.shadow_boost_val = 2.0
        
        # Parámetros de Clustering
        self.cluster_eps = 0.5
        self.cluster_min_samples = 5
        self.cluster_min_pts = 10
        
        # Parámetros del Hull (Solo Cóncavo)
        self.hull_alpha = 0.1
        self.hull_vis_z = -1.7
        
        # Inicialización del Estado (Memoria Temporal y Consistencia)
        # -------------------------------------------------------------------------
        # Uso en el código:
        # 1. current_scan/frame_count: Control del Flujo. Indican qué archivo .bin cargar y el progreso.
        #    Sin esto, el sistema no sabría qué datos procesar en cada iteración.
        #
        # 2. prev_abs_range: "Depth Jump Check" (Seguro contra Ghosting).
        #    Almacena las distancias reales (rango absoluto) del frame anterior.
        #    CRÍTICO: Al mover la memoria (warp) al nuevo frame, comparamos si la distancia guardada coincide con la nueva.
        #    Si hay un salto brusco (>0.2m), el sistema entiende que el objeto se movió o es uno nuevo (ej. peatón que pasa),
        #    y reinicia la memoria en ese píxel para no "arrastrar" sombras viejas ni mezclar objetos distintos.
        #
        # 3. belief_map: Memoria de Probabilidad (Log-Odds).
        #    Matriz 64x2048 que almacena el "nivel de creencia" acumulado de obstáculo.
        #    Permite que la detección NO PARPADEE. Si un obstáculo se ve en el frame 1, se recuerda en el 2.
        #    El ruido aleatorio (polvo, hierba) se filtra porque no es consistente en el tiempo.
        #
        # 4. poses: Odometría (Compensación de Ego-Motion).
        #    Lista de matrices 4x4 con la posición global del coche en cada instante.
        #    Necesario para "mover" el mapa de creencia (warp) y que los objetos estáticos se mantengan en su sitio
        #    aunque el coche se mueva. Sin esto, los objetos dejarían estelas o "fantasmas".
        # -------------------------------------------------------------------------
        self.current_scan = self.scan_start
        self.frame_count = 0
        self.prev_abs_range = None
        self.belief_map = np.zeros((self.H, self.W), dtype=np.float64)
        self.poses = self.load_poses()
        self.current_pose = np.eye(4)
        self.gt_semantic = None
        self.has_gt = False
        self.prev_hull = None  # Hull del frame anterior para limitar sombras
        
        # Calibración (Tr_velo_to_cam)
        # -------------------------------------------------------------------------
        # Matriz de transformación extrínseca fija entre LiDAR (Velodyne) y Cámara (Referencia KITTI).
        # USO CRÍTICO: Las poses de KITTI suelen estar en coordenadas de cámara.
        # Usamos `Tr` para convertir esas poses al sistema del Velodyne.
        # Sin esta matriz, al intentar compensar el movimiento, los puntos se proyectarían en lugares erróneos
        # (ej. el suelo se inclinaría o los edificios se verían desplazados), rompiendo la consistencia temporal.
        # -------------------------------------------------------------------------
        self.Tr = np.array([
            [4.2768028e-04, -9.9996725e-01, -8.0844917e-03, -1.1984599e-02],
            [-7.2106265e-03, 8.0811985e-03, -9.9994132e-01, -5.4039847e-02],
            [9.9997386e-01, 4.8594858e-04, -7.2069002e-03, -2.9219686e-01],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Inicializar Submódulos
        self.init_patchwork()
        self.cv_bridge = CvBridge()
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()
        
        # Inicializar PublicadoresEstoy desarrollando un sistema de percepción LiDAR 3D para entornos off-road (TFG) enfocado en la detección de obstáculos positivos, negativos (hundimientos/baches) y anomalías de visibilidad (Voids) utilizando el dataset GOose. Mi pipeline actual utiliza Patchwork++ para la segmentación de suelo y una proyección de imagen de rango para el análisis de anomalías geométricas mediante la métrica $\Delta r$ (Rango esperado vs. medido). He realizado una auditoría del código fuente de Patchwork++ y he descubierto un fallo de seguridad crítico: el algoritmo acepta planos verticales (paredes) como suelo en las zonas 1 a 3 ($>9.64m$) porque su filtro de verticalidad RVPF solo está activo en la zona 0. Mi objetivo es implementar una Capa de Validación de Integridad basada en la normal vertical ($n_z$) y varianza local, inspirada en la metodología de ROBIO 2024 (que usa filtros de altura $T_{height}$ y varianza $T_{var}$), para evitar falsos negativos en el horizonte. Actualmente, el sistema procesa cada frame en 197 ms sobre un hardware i7-1255U, siendo Patchwork++ el cuello de botella (147 ms), por lo que busco optimizar la lógica y robustecer la detección de 'voids' y obstáculos negativos sin penalizar más la latencia.
        self.init_publishers()
        
        self.get_logger().info(f"ObjectsDetector Inicializado. Escena: {scene}, Scans: {scan_start}-{scan_end}")
        
        # Ejecutar Procesamiento por Lotes
        self.run_batch_processing()
        
        # Timer para republicar visualización
        self.timer = self.create_timer(1.0, self.publish_visualization)

    def init_publishers(self):
        """Inicializa todos los publicadores ROS2."""
        self.range_image_pub = self.create_publisher(Image, 'range_image', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, 'point_cloud', 10)
        self.delta_r_cloud_pub = self.create_publisher(PointCloud2, 'delta_r_cloud', 10)
        self.delta_r_filtered_pub = self.create_publisher(PointCloud2, 'delta_r_filtered_cloud', 10)
        self.ground_cloud_pub = self.create_publisher(PointCloud2, 'ground_cloud', 10)
        self.gt_cloud_pub = self.create_publisher(PointCloud2, 'gt_cloud', 10)
        self.bayes_cloud_pub = self.create_publisher(PointCloud2, 'bayes_cloud', 10)
        self.bayes_temporal_pub = self.create_publisher(PointCloud2, 'bayes_temporal_cloud', 10)
        self.cluster_points_pub = self.create_publisher(PointCloud2, 'cluster_points', 10)
        
        # Sombras y Huecos (Voids)
        self.shadow_pub = self.create_publisher(Marker, 'geometric_shadows', 10)
        self.text_pub = self.create_publisher(MarkerArray, 'shadow_text_markers', 10)
        self.shadow_cloud_pub = self.create_publisher(PointCloud2, 'shadow_cloud', 10)
        self.pub_voids = self.create_publisher(PointCloud2, 'void_cloud', 10)
        
        # Hull (Solo Cóncavo)
        self.hull_pub = self.create_publisher(Marker, 'concave_hull', 10)

    def broadcast_static_tf(self):
        """Publica una transformada estática identidad entre map y velodyne."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'velodyne'
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def init_patchwork(self):
        """Inicializa la librería Patchwork++ para segmentación de suelo."""
        try:
            import pypatchworkpp
            self.params = pypatchworkpp.Parameters()
            self.params.verbose = False
            self.params.sensor_height = self.default_ground_height # Se usa como estimación inicial de la altura del sensor para que Patchwork++ sepa dónde buscar el plano base.
            self.params.min_range = self.min_range
            self.params.max_range = self.max_range
            self.params.num_iter = 3
            self.params.num_lpr = 20
            self.params.num_min_pts = 10
            self.params.th_dist = 0.2
            self.params.uprightness_thr = 0.707
            self.params.adaptive_seed_selection_margin = -1.1
            self.params.enable_RNR = False
            
            # Parámetros CZM (Concentric Zone Model)
            # -------------------------------------------------------------------------
            # Patchwork++ divide el espacio alrededor del coche en 4 zonas concéntricas (CZM).
            # En cada zona, ajusta múltiples planos de suelo independientes para adaptarse a pendientes.
            # 1. num_zones: 4 anillos principales de distancia.
            # 2. num_rings_each_zone: Cuántos sub-anillos radiales tiene cada zona (más detalle cerca).
            # 3. num_sectors_each_zone: Cuántos "quesitos" o sectores angulares tiene cada zona.
            #    Zona 3 tiene 54 sectores para máxima precisión a media distancia.
            # -------------------------------------------------------------------------
            self.params.num_zones = 4
            self.params.num_rings_each_zone = [2, 4, 4, 4]
            self.params.num_sectors_each_zone = [16, 32, 54, 32]

            self.patchwork = pypatchworkpp.patchworkpp(self.params)
            self.initialize_czm_params()
        except ImportError:
            self.get_logger().error("No se pudo importar pypatchworkpp.")
            sys.exit(1)

    def initialize_czm_params(self):
        """Calcula los tamaños de anillos y sectores para el modelo de zonas concéntricas."""
        min_r = self.params.min_range
        max_r = self.params.max_range
        self.min_ranges = [min_r, (7*min_r+max_r)/8.0, (3*min_r+max_r)/4.0, (min_r+max_r)/2.0]
        self.ring_sizes = []
        for i in range(4):
            end_r = self.min_ranges[i+1] if i < 3 else max_r
            self.ring_sizes.append((end_r - self.min_ranges[i]) / self.params.num_rings_each_zone[i])
        self.sector_sizes = [2 * np.pi / n for n in self.params.num_sectors_each_zone]

    def run_batch_processing(self):
        """Ejecuta el procesamiento de todos los frames definidos en scan_start y scan_end."""
        t0 = _time.time()
        for scan_i in range(self.scan_start, self.scan_end + 1):
            self.current_scan = scan_i
            self.process_frame()
            self.frame_count += 1
        elapsed = _time.time() - t0
        self.get_logger().info(f"Procesamiento por lotes completado: {self.frame_count} frames en {elapsed:.1f}s")
        
        # Publicar el último frame (con acumulación Bayes completa) en RViz
        self.publish_visualization()

    def process_frame(self):
        """
        Orquesta el pipeline principal de procesamiento para un solo frame.
        Pasos:
        1. Cargar Datos
        2. Segmentación de Suelo
        3. Proyección de Rango
        4. Filtro Bayesiano (Temporal)
        5. Filtro Espacial (Inter-Ring)
        6. Clustering
        7. Cálculo de Hull Cóncavo
        8. Generación de Visualizaciones
        """
        t_start = _time.time()
        
        # Reiniciar acumulaciones
        self.shadow_text_markers = MarkerArray()
        
        # 1. Cargar Datos
        points, remissions = self.load_current_scan()
        if points is None: return

        # 2. Segmentación de Suelo
        ground_points, n_per_point, d_per_point, rejected_mask = self.segment_ground(points)
        self.rejected_mask = rejected_mask # Guardar para uso en Bayes
        
        # 3. Proyección de Rango
        range_image, delta_r, u, v, r, r_exp = self.compute_range_projection(points, n_per_point, d_per_point)
        self.current_range_image = range_image # Guardar para el siguiente frame
        
        # 4. Filtrar Objetos (Bayes Temporal)
        P_belief, P_per_point_temporal = self.run_bayes_filter(points, range_image, delta_r, u, v, d_per_point)
        
        # 5. Filtro Espacial Inter-Ring (Paso Separado)
        P_final = self.apply_inter_ring_filter(P_belief)
        P_per_point = P_final[u, v]
        
        # 6. Clustering
        clusters, cluster_cloud_msg = self.run_clustering(points, P_per_point, u, v)
        self.detected_clusters = clusters # Guardar para Sombra/Hull
        self.cluster_points_msg = cluster_cloud_msg
        
        # 7. Hull Cóncavo (Sin Convexo)
        self.compute_concave_hull(points)
        
        # 8. Generación de Visualizaciones
        self.generate_visualizations(points, range_image, delta_r, P_per_point, ground_points, n_per_point, u, v)
        
        t_end = _time.time()
        # self.get_logger().info(f"Frame {self.current_scan} Procesado en {(t_end-t_start)*1000:.1f}ms")

    # =========================================================================
    # 1. CARGA DE DATOS
    # =========================================================================
    def load_current_scan(self):
        """Carga los puntos del escaneo actual (.bin) y la pose si está disponible."""
        scan_str = f"{int(self.current_scan):06d}"
        bin_path = self.data_path / "sequences" / self.scene / "velodyne" / f"{scan_str}.bin"
        if not bin_path.exists():
            # Intentar ruta alternativa
            bin_path = self.data_path / "velodyne" / f"{scan_str}.bin"
            if not bin_path.exists():
                self.get_logger().error(f"Archivo bin no encontrado: {bin_path}")
                return None, None
                
        # Cargar Puntos
        scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        remissions = scan[:, 3]
        
        # Cargar Pose
        if self.poses is not None and len(self.poses) > self.current_scan:
            p = self.poses[self.current_scan]
            self.current_pose = np.vstack([p, [0,0,0,1]]) if p.shape == (3,4) else p
        
        # Cargar Etiquetas (Opcional)
        self.load_labels(scan_str)
        
        return points, remissions

    def load_labels(self, scan_str):
        """Carga las etiquetas semánticas Ground Truth (.label) si existen."""
        label_root = self.data_path.parent.parent / 'data_odometry_labels' / 'dataset'
        label_path = label_root / 'sequences' / self.scene / 'labels' / f"{scan_str}.label"
        if label_path.exists():
            self.gt_semantic = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
            self.has_gt = True
        else:
            self.has_gt = False

    def load_poses(self):
        """Carga el archivo poses.txt de la secuencia."""
        candidates = [
            self.data_path / 'sequences' / self.scene / 'poses.txt',
            self.data_path.parent.parent / 'data_odometry_labels' / 'dataset' / 'sequences' / self.scene / 'poses.txt',
        ]
        for p in candidates:
            if p.exists():
                poses = []
                with open(p, 'r') as f:
                    for line in f:
                        vals = np.array([float(v) for v in line.split()]).reshape(3, 4)
                        poses.append(vals)
                return poses
        return None

    # =========================================================================
    # 2. SEGMENTACIÓN DE SUELO
    # =========================================================================
    def segment_ground(self, points):
        """
        Ejecuta todo el pipeline de segmentación de suelo.
        FLUJO DE EJECUCIÓN:
        1. Patchwork++: Estima puntos de suelo "brutos" y calcula planos locales para cada zona/anillo/sector (CZM).
        2. analyze_local_planes: Revisa cada plano local devuelto. Si su pendiente es muy alta (pared) 
           o su rugosidad es excesiva (vegetación), lo marca como inválido.
        3. compute_global_plane: Calcula un plano promedio global (SVD) por si acaso fallan los locales.
        4. assign_planes_to_points: Para CADA punto del escaneo, busca en qué bin CZM cae y le asigna 
           la ecuación del plano (normal n, distancia d) correspondiente a esa zona.
           Si el bin no tiene plano válido, usa el global.
        5. create_rejected_mask: Genera una máscara booleana para ignorar puntos que caen en bins detectados como paredes/ruido. 
        """
        # Ejecutar Patchwork++ (Librería externa en C++)
        self.patchwork.estimateGround(points)
        ground_points = self.patchwork.getGround()
        
        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()
        
        # Lógica de Planos Locales
        local_planes, rejected_bins, rejected_cents = self.analyze_local_planes(centers, normals, ground_points)
        
        # Fallback Global
        global_n, global_d = self.compute_global_plane(ground_points)
        
        # Asignar n, d a cada punto
        n_per_point, d_per_point = self.assign_planes_to_points(points, local_planes, global_n, global_d)
        
        # Crear máscara de rechazados (puntos en bins de pared)
        rejected_mask = self.create_rejected_mask(points, rejected_bins)
        
        return ground_points, n_per_point, d_per_point, rejected_mask

    def analyze_local_planes(self, centers, normals, ground_points):
        """
        Analiza los planos devueltos por Patchwork++ para rechazar paredes o vegetación.
        Flujo de Decisión:
        1. Identifica el bin CZM de cada centro de plano.
        2. Chequeo de Pendiente: Si normal.z < 0.7 -> Pared.
        # 3. Chequeo de Rugosidad (opcional si hay 'tree'):
        #    'tree' es un KD-Tree (estructura de búsqueda en el espacio).
        #    VENTAJA CRITICA: Nos permite buscar vecinos reales en un radio (0.5m)
        #    y calcular la varianza de altura (Z_max - Z_min).
        #    Sin esto, solo tendríamos la normal del plano matemático, que puede ser engañosa
        #    en hierba alta o rocas. Con el tree, medimos la "textura" física del suelo.
        #    Si dz > 0.3 -> Es superficie rugosa (vegetación/piedras), NO transitable.
        Retorna un dict con los planos válidos: {bin_id: (normal, distancia)}.
        """
        local_planes = {} # (z,r,s) -> (n, d)
        rejected_bins = set()
        rejected_centroids = []
        
        has_tree = False
        if len(ground_points) > 0:
            try:
                tree = cKDTree(ground_points)
                has_tree = True
            except: pass
            
        for i, (c, n) in enumerate(zip(centers, normals)):
            bin_id = self.get_czm_bin_scalar(c[0], c[1])
            is_wall = False
            
            # Rechazo de Pared basado en Pendiente (Slope-Aware Wall Rejection)
            # Si la normal es muy horizontal (componente Z pequeña), es probablemente una pared, no suelo.
            if abs(n[2]) < self.wall_rejection_slope:
                if has_tree:
                    idx = tree.query_ball_point(c, 0.5)
                    if len(idx) > 5:
                        pts = ground_points[idx]
                        dz = np.percentile(pts[:,2], 95) - np.percentile(pts[:,2], 5)
                        # Si la variación de altura en un parche pequeño es alta, es vegetación/obstáculo, no suelo plano.
                        if dz > self.wall_height_diff_threshold:
                            is_wall = True
                    else:
                        if c[2] > -1.0: is_wall = True
                else:
                    if c[2] > -1.0: is_wall = True
            
            if is_wall:
                if bin_id: rejected_bins.add(bin_id)
                rejected_centroids.append(c)
                continue
                
            if n[2] < 0: n = -n
            if bin_id:
                local_planes[bin_id] = (n, -np.dot(n, c))
                
        return local_planes, rejected_bins, rejected_centroids

    def compute_global_plane(self, ground_points):
        """Calcula un plano global usando SVD como fallback si no hay planos locales válidos."""
        if len(ground_points) > 10:
            centroid = np.mean(ground_points, axis=0)
            u, s, vt = np.linalg.svd(ground_points - centroid, full_matrices=False)
            normal = vt[2, :]
            if normal[2] < 0: normal = -normal
            d = -np.dot(normal, centroid)
            return normal, d
        return np.array([0,0,1]), 1.73

    def assign_planes_to_points(self, points, local_planes, global_n, global_d):
        """
        Asigna a cada punto del escaneo su normal y distancia al suelo correspondiente.
        Lógica:
        1. Vectoriza el cálculo de bins (z, r, s) para todos los puntos.
        2. Crea una tabla de consulta (LUT) 4D con los planos locales válidos.
        3. Rellena los huecos con el plano global (por defecto).
        4. Devuelve arrays n_out, d_out alineados con la nube de puntos original.
        """
        # Búsqueda Vectorizada
        z_idx, r_idx, s_idx = self.get_czm_bin_vectorized(points[:,0], points[:,1])
        
        # 4 zonas, 4 anillos max, 54 sectores max
        table = np.zeros((4, 4, 54, 4), dtype=np.float32)
        table[..., :3] = global_n
        table[..., 3] = global_d
        
        for (z,r,s), (n, d) in local_planes.items():
            if 0<=z<4 and 0<=r<4 and 0<=s<54:
                table[z,r,s, :3] = n
                table[z,r,s, 3] = d
                
        valid = (z_idx>=0) & (r_idx>=0) & (s_idx>=0)
        n_out = np.full((len(points), 3), global_n, dtype=np.float32)
        d_out = np.full(len(points), global_d, dtype=np.float32)
        
        if np.any(valid):
            params = table[z_idx[valid], r_idx[valid], s_idx[valid]]
            n_out[valid] = params[:, :3]
            d_out[valid] = params[:, 3]
            
        return n_out, d_out

    def create_rejected_mask(self, points, rejected_bins):
        """
        Crea una máscara booleana True/False para puntos que caen en bins rechazados.
        Si un punto cae en una zona marcada como pared en `analyze_local_planes`,
        esta función lo marca para que NO sea considerado suelo en pasos posteriores.
        """
        z, r, s = self.get_czm_bin_vectorized(points[:,0], points[:,1])
        mask = np.zeros(len(points), dtype=bool)
        for (zb, rb, sb) in rejected_bins:
            mask |= (z == zb) & (r == rb) & (s == sb)
        return mask

    def get_czm_bin_scalar(self, x, y):
        """Retorna el bin (zona, anillo, sector) para un punto dado (Escalar)."""
        r = np.sqrt(x*x + y*y)
        if r <= self.params.min_range or r > self.params.max_range: return None
        theta = np.arctan2(y, x)
        if theta < 0: theta += 2*np.pi
        
        for z in range(4):
            r_end = self.min_ranges[z+1] if z < 3 else self.params.max_range
            if r < r_end:
                r_start = self.min_ranges[z]
                r_idx = int((r - r_start) / self.ring_sizes[z])
                r_idx = min(r_idx, self.params.num_rings_each_zone[z]-1)
                
                s_idx = int(theta / self.sector_sizes[z])
                s_idx = min(s_idx, self.params.num_sectors_each_zone[z]-1)
                return (z, r_idx, s_idx)
        return None

    def get_czm_bin_vectorized(self, x, y):
        """Versión vectorizada para obtener bins de múltiples puntos a la vez."""
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2*np.pi
        
        z_idx = np.full_like(r, -1, dtype=np.int32)
        r_idx = np.full_like(r, -1, dtype=np.int32)
        s_idx = np.full_like(r, -1, dtype=np.int32)
        
        valid = (r > self.params.min_range) & (r <= self.params.max_range)
        
        for z in range(4):
            r_start = self.min_ranges[z]
            r_end = self.min_ranges[z+1] if z < 3 else self.params.max_range
            mask = valid & (r >= r_start) & (r < r_end)
            if np.any(mask):
                z_idx[mask] = z
                r_idx[mask] = ((r[mask] - r_start) / self.ring_sizes[z]).astype(np.int32)
                s_idx[mask] = (theta[mask] / self.sector_sizes[z]).astype(np.int32)
                
                # Clip
                r_idx[mask] = np.clip(r_idx[mask], 0, self.params.num_rings_each_zone[z]-1)
                s_idx[mask] = np.clip(s_idx[mask], 0, self.params.num_sectors_each_zone[z]-1)
                
        return z_idx, r_idx, s_idx

    # =========================================================================
    # 3. PROYECCIÓN DE RANGO
    # =========================================================================
    def compute_range_projection(self, points, n, d):
        """
        Calcula la "Imagen de Rango" (Range Image).
        Lógica:
        1. Para cada punto, conocemos su dirección (vector unitario) y el plano de suelo (n, d) debajo.
        2. Calculamos R_esperado: ¿A qué distancia debería estar el suelo en esa dirección?
           Fórmula: R_exp = -d / (n · dirección)
        3. Calculamos Delta R: Diferencia entre lo que mide el sensor (R) y lo esperado (R_exp).
           - Delta R ~ 0: Es suelo plano.
           - Delta R < 0 (ej. -2m): El punto está 2m ANTES de lo esperado -> OBSTÁCULO.
        4. Proyectamos todo a una imagen 2D (u, v) para procesarlo eficientemente luego.
        """
        r = np.linalg.norm(points, axis=1)
        
        # Producto punto
        dot = np.sum(points * n, axis=1)
        valid_dot = dot < -1e-3
        
        r_exp = np.full(len(points), 999.9, dtype=np.float32)
        r_exp[valid_dot] = -d[valid_dot] * r[valid_dot] / dot[valid_dot]
        
        delta_r = r - r_exp
        delta_r = np.clip(delta_r, -20.0, 10.0)
        
        # Proyectar a Imagen
        u, v, valid_fov = self.project_points_to_uv(points, r)
        
        # Construir Imagen
        range_image = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Ordenar por rango (descendente) para que los puntos más cercanos sobrescriban a los lejanos 
        # (Rendering estilo "Z-Buffer" inverso: queremos ver lo más cercano en cada pixel)
        order = np.argsort(r)[::-1]
        u_s, v_s, d_s = u[order], v[order], delta_r[order]
        
        range_image[u_s, v_s] = d_s
        
        return range_image, delta_r, u, v, r, r_exp

    def project_points_to_uv(self, points, r=None):
        """
        Proyección Esférica: Convierte puntos 3D (x,y,z) a coordenadas de imagen 2D (u,v).
        Usa el modelo de cámara "pinhole" esférico para el Velodyne.
        - u (filas): Relacionado con el ángulo vertical (pitch) y los anillos del láser.
        - v (columnas): Relacionado con el ángulo horizontal (yaw) o azimut.
        """
        if r is None: r = np.linalg.norm(points, axis=1)
        valid = r > 0.1
        
        # Pitch: Ángulo de elevación
        pitch = np.arcsin(np.clip(points[:,2]/r, -1, 1))
        # Yaw: Ángulo de azimut
        yaw = np.arctan2(points[:,1], points[:,0])
        
        # Mapeo lineal de Pitch -> u (filas 0..63)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        u = np.floor(proj_y * self.H).astype(np.int32)
        u = np.clip(u, 0, self.H - 1)
        
        # Mapeo lineal de Yaw -> v (columnas 0..2047)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        v = np.floor(proj_x * self.W).astype(np.int32)
        v = np.clip(v, 0, self.W - 1)
        
        return u, v, valid

    # =========================================================================
    # 4. FILTRO BAYESIANO
    # =========================================================================
    def run_bayes_filter(self, points, range_image, delta_r, u, v, d_per_point):
        """
        Ejecuta el filtro Bayesiano temporal (Binary Bayes Filter).
        ESTRATEGIA:
        1. Convertimos la medición actual (Delta R) a probabilidades "Raw".
           - Si Delta R < umbral -> Probabilidad alta de obstáculo.
        2. "Warp": Traemos la memoria del pasado (t-1) al presente (t) compensando el movimiento del coche.
        3. Fusión: Combinamos la memoria movida con la nueva medición usando Log-Odds.
           - Log-Odds permite sumar probabilidades: L_t = L_t-1 + L_meas - L_0
        4. "Shadow Boost": Si detectamos sombras, incrementamos la certeza artificialmente.
        5. Convertimos Log-Odds de vuelta a probabilidad (0..1) para usarla fuera.
        """
        # 1. Probabilidad Raw (Modelo del Sensor Inverso Simplificado)
        # Si range_image < threshold (-0.3m), es obstáculo (1.0), si no suelo (0.0)
        P_raw = (range_image < self.threshold_obs).astype(np.float32)
        
        # 2. Warp y Actualización de Creencia
        # Aquí ocurre la magia de la memoria temporal
        P_belief, self.belief_map = self.update_belief(self.belief_map, P_raw, points, u, v)
        
        # 3. Aplicar Boost de Sombras
        # Si hay "vacío" detrás de un obstáculo, confirmamos que es sólido sumando valor a su creencia
        shadow_boost, voids = self.detect_geometric_shadows(range_image, u, v, points, d_per_point)
        # Guardar voids para visualización
        self.void_points = voids
        
        self.belief_map += shadow_boost
        # Clamping para evitar que la creencia crezca infinitamente (saturación)
        np.clip(self.belief_map, self.belief_clamp_min, self.belief_clamp_max, out=self.belief_map)
        
        # Recuperar probabilidad normalizada (Sigmoide)
        P_belief = 1.0 / (1.0 + np.exp(-self.belief_map))
        
        # 4. Mapear la probabilidad de la imagen 2D a cada punto 3D original
        P_per_point = P_belief[u, v]
        
        # Guardar imagen de rango actual para el "Depth Jump Check" del siguiente frame
        self.prev_abs_range = self.current_range_image.copy()
        
        return P_belief, P_per_point

    def update_belief(self, belief_map, P_obs, points, u_curr, v_curr):
        """
        Actualiza el mapa de creencia Bayesiano usando Log-Odds.
        Fórmula: L_t = Warped(L_t-1) + Log(P_meas / (1-P_meas)) - L_0
        """
        # A. Warp: Mover memoria del pasado al presente
        warped_belief = self.warp_belief_map(belief_map, points, u_curr, v_curr)
        
        # B. Calcular Log-Odds de la medición actual (Inverse Sensor Model)
        eps = 1e-6
        P_clamped = np.clip(P_obs, eps, 1.0 - eps) # Evitar log(0)
        l_obs = np.log(P_clamped / (1.0 - P_clamped))
        
        # C. Suma Recursiva (Bayes)
        new_map = warped_belief + l_obs - self.l0
        new_map = np.clip(new_map, self.belief_clamp_min, self.belief_clamp_max)
        
        P_out = 1.0 / (1.0 + np.exp(-new_map))
        return P_out, new_map

    def warp_belief_map(self, belief_map, points_curr, u_curr, v_curr):
        """
        Transforma (warp) el mapa de creencia anterior al frame actual usando la odometría.
        PASOS:
        1. Calcula la transformación relativa T_rel entre scan(t-1) y scan(t).
        2. Proyecta los puntos ACTUALES hacia atrás en el tiempo: P_prev = T_rel_lidar * P_curr.
        3. Calcula dónde caían esos puntos en la imagen anterior (u_prev, v_prev).
        4. Lee el valor de creencia antiguo en esas coordenadas y lo suma a la posición actual.
        """
        if self.frame_count == 0 or self.poses is None: return belief_map
        
        # Obtener T_rel (Movimiento Relativo)
        tc = self.poses[self.current_scan]
        if self.current_scan > 0:
            tp = self.poses[self.current_scan - 1]
        else: return belief_map
        
        T_rel = np.linalg.inv(tp) @ tc
        # Frame Lidar
        T_rel_lidar = np.linalg.inv(self.Tr) @ T_rel @ self.Tr
        
        # Transformar puntos al frame previo
        hom = np.hstack([points_curr, np.ones((len(points_curr), 1))])
        pts_prev = (T_rel_lidar @ hom.T).T[:, :3]
        
        u_prev, v_prev, valid = self.project_points_to_uv(pts_prev)
        
        warped = np.zeros_like(belief_map)
        
        # Comprobación de Salto de Profundidad (Depth Jump Check)
        mask = valid.copy()
        if self.prev_abs_range is not None:
             r_sens_prev = self.prev_abs_range[u_prev[valid], v_prev[valid]]
             r_pt_prev = np.linalg.norm(pts_prev[valid], axis=1)
             # DEPTH JUMP CHECK: Verificar que la geometría sea consistente (< 0.2m de diferencia)
             mask[valid] &= (np.abs(r_sens_prev - r_pt_prev) < self.depth_jump_threshold)
        
        # Mapear valores antiguos a nuevas coordenadas
        if np.any(mask):
            old_vals = belief_map[u_prev[mask], v_prev[mask]]
            np.add.at(warped, (u_curr[mask], v_curr[mask]), old_vals)
            
        return warped

    def apply_inter_ring_filter(self, P):
        """
        Aplica un filtro espacial horizontal (suavizado inter-anillo) para asegurar consistencia.
        - Es una Convolución 1D con Kernel [0.25, 0.5, 0.25].
        - Es CIRCULAR (usando np.roll): Los bordes de la imagen (izq/der) se conectan porque el LiDAR gira 360º.
        Fórmula: 0.5 * Centro + 0.25 * Izq + 0.25 * Der
        """
        mu = 0.5 # Factor de mezcla
        # np.roll es circular: el último pixel se conecta con el primero.
        P_left = np.roll(P, 1, axis=1)
        P_right = np.roll(P, -1, axis=1)
        return (1-mu)*P + mu*(P_left + P_right)/2.0

    # =========================================================================
    # 5. DETECCIÓN DE SOMBRAS (GRANULAR)
    # =========================================================================
    def detect_geometric_shadows(self, range_image, u_all, v_all, points, d_per_point):
        """
        Detecta sombras geométricas y vacíos (voids).
        Retorna mapa de boost de sombras y puntos de vacíos.
        """
        # 1. Mapa de Altura
        height_map, d_img, pitch_map, sin_neg_p = self.compute_shadow_height_map(u_all, v_all, points, d_per_point)
        
        # 2. Vacíos (Obstáculos Negativos)
        void_boost, void_points = self.detect_voids(d_img, sin_neg_p, height_map, range_image, pitch_map)
        
        # 3. Sombras Proyectadas (Cast Shadows)
        shadow_boost = self.compute_shadow_casting(range_image, height_map, d_img, sin_neg_p)
        
        # Guardar el mapa de boost de sombras para visualización (ANTES de escalar)
        self.shadow_boost_map = shadow_boost.copy()
        self.shadow_pitch_map = pitch_map
        
        return (void_boost + shadow_boost) * self.shadow_boost_val, void_points

    def compute_shadow_height_map(self, u, v, points, d):
        """Pre-calcula mapas de altura y geometría para detección rápida de sombras."""
        d_img = np.full((self.H, self.W), 1.73, dtype=np.float32)
        d_img[u, v] = d
        
        z_img = np.full((self.H, self.W), -999.0, dtype=np.float32)
        z_img[u, v] = points[:, 2]
        
        height_map = z_img + d_img
        
        u_idx = np.arange(self.H).reshape(-1, 1)
        pitch_map = np.radians((1.0 - u_idx/self.H)*28.0 - 25.0)
        sin_neg_p = np.sin(-pitch_map)
        
        return height_map, d_img, pitch_map, sin_neg_p

    def detect_voids(self, d_img, sin_p, height_map, range_image, pitch_map):
        """
        Detecta 'vacíos' donde se espera suelo pero el sensor no ve nada.
        Indica posibles obstáculos absorbentes o agujeros.
        """
        valid_proj = sin_p > 0.08
        r_exp = np.full((self.H, self.W), 999.0, dtype=np.float32)
        np.divide(d_img, sin_p, out=r_exp, where=valid_proj)
        
        mask_valid = (range_image > 0.5) & (range_image < 60.0)
        # Vacío: Se espera suelo cerca (<15m) pero sensor no ve NADA
        mask_void = (~mask_valid) & valid_proj & (r_exp < 15.0)
        
        boost = np.zeros((self.H, self.W), dtype=np.float32)
        boost[mask_void] = 0.8
        
        # Reconstruir puntos de Vacío para visualización
        void_pts = []
        if np.any(mask_void):
            u, v = np.where(mask_void)
            r = r_exp[mask_void]
            p = pitch_map[u, 0]
            y = (2.0*v/float(self.W) - 1.0)*np.pi
            
            rcos = r * np.cos(p)
            x = rcos * np.cos(y)
            y = rcos * np.sin(y)
            z = r * np.sin(p)
            void_pts = np.column_stack((x,y,z))
            
        return boost, void_pts

    def compute_shadow_casting(self, range_image, height_map, d_img, sin_p_map):
        """Calcula sombras proyectadas iterando por columnas."""
        boost = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Iterar columnas
        for v in range(self.W):
            self.scan_column_for_shadows(v, range_image[:,v], height_map[:,v], d_img[:,v], sin_p_map[:,0], boost)
            
        # El 'boost' aquí es una matriz que contiene incrementos de probabilidad
        # donde se ha detectado una sombra geométrica clara.
        return boost

    def scan_column_for_shadows(self, v, col_r, col_h, col_d, col_sin, boost_map):
        """
        Máquina de estados para escanear una columna vertical y detectar sombras detrás de objetos.
        Estado 0: Buscar objeto.
        Estado 1: Seguir objeto.
        Estado 2: Validar sombra.
        """
        state = 0 # 0=Buscar, 1=Seguir, 2=Validar
        obs_r = -1.0
        obs_h = -999.0
        obs_u = -1
        shadow_end = 0.0
        
        for u in range(self.H - 1, -1, -1):
            r_val = col_r[u]
            is_valid = (r_val > 0.5) and (r_val < 60.0)
            is_obj = is_valid and (col_h[u] > 0.2)
            
            if state == 0:
                if is_obj:
                    state = 1
                    obs_r = r_val
                    obs_h = col_h[u]
                    obs_u = u
            elif state == 1:
                if is_valid and abs(r_val - obs_r) < 1.0:
                    obs_h = max(obs_h, col_h[u])
                    obs_u = u
                    obs_r = min(obs_r, r_val)
                else:
                    # Fin del objeto -> Calcular longitud de sombra teórica (L)
                    # Geometría: L = Altura * (R_obj / (H_sensor - H_obj))
                    # L es cuanto se extiende la sombra en el plano horizontal.
                    L = obs_r * obs_h / max(0.1, col_d[obs_u] - obs_h)
                    
                    # shadow_end: Hasta dónde comprobamos si hay vacío.
                    # Se limita L a máximo 50 metros para no extender sombras infinitas en el horizonte.
                    shadow_end = obs_r + min(L, 50.0)
                    
                    state = 2
                    # Pasar a validar este pixel inmediatamente
            
            if state == 2:
                # Comprobar Sombra
                sin_val = col_sin[u]
                r_gnd = col_d[u] / sin_val if sin_val > 0.01 else 999.0
                
                # ¿Estamos DENTRO de la longitud de sombra esperada?
                # Si r_gnd < shadow_end: El punto del suelo que estamos mirando DEBERÍA estar a la sombra del objeto.
                # Es aquí donde verificamos la hipótesis:
                # - Si está vacío -> Confirmamos objeto sólido.
                # - Si vemos suelo -> El objeto no tapó el láser (fantasma/transparente).
                # Si r_gnd > shadow_end: Ya estamos demasiado lejos, la sombra teórica acabó.
                if r_gnd < shadow_end:
                    if not is_valid:
                        # -----------------------------------------------------------------
                        # CALCULO DEL BOOST DE SOMBRA
                        # 1. Si estamos en zona teórica de sombra (detrás del objeto)...
                        # 2. Y el láser NO ve nada (is_valid es False, o sea "vacío")...
                        # 3. Entonces CONFIRMAMOS que el objeto es sólido (está bloqueando la luz).
                        #
                        # factor: Confianza basada en distancia del objeto (obs_r).
                        # - Cerca (0m): factor = 1.0
                        # - Lejos (60m): factor -> 0.0 (Decae linealmente)
                        # - Mínimo: 0.2 (Siempre confiamos un poco incluso lejos)
                        #
                        # boost: Sumamos +0.5 * factor al mapa.
                        # -----------------------------------------------------------------
                        factor = max(0.2, 1.0 - obs_r/60.0)
                        boost_map[obs_u, v] += 0.5 * factor
                    elif abs(r_val - r_gnd) < 0.5:
                        # Suelo detectado -> Supresión (No hay sombra)
                        # Si vemos suelo justo donde debería haber sombra, el objeto quizás era fantasma o transparente.
                        # Restamos probabilidad.
                        boost_map[obs_u, v] -= 0.2
                    elif is_obj:
                        # EDGE CASE: Objeto B dentro de la sombra teórica de Objeto A.
                        # - El Filtro Bayesiano (P_raw) SÍ lo detectará como obstáculo (por su Rango).
                        # - Pero la lógica de sombras NO le calculará su propia sombra (estamos ocupados validando A).
                        # - Tampoco confirma ni niega a A. Simplemente se "abstiene" de votar.
                        #   (No sumamos ni restamos boost).
                        pass
                else:
                    # HEMOS SALIDO DE LA SOMBRA (r_gnd > shadow_end)
                    # Hemos chequeado todos los pixeles desde el objeto hasta el fin de la sombra teórica.
                    # El 'boost_map[obs_u, v]' ahora contiene la SUMA ACUMULADA de evidencia:
                    # +0.5 por cada pixel vacío (confirma sólido)
                    # -0.2 por cada pixel de suelo (contradice sólido)
                    state = 0
                    # RE-ENTRADA INMEDIATA: "Cadena de Objetos"
                    # Acabamos de salir de la sombra del Objeto A.
                    # Pero el pixel actual `u` podría ser el inicio del Objeto B situado detrás.
                    # Si no hacemos este chequeo, perderíamos el Objeto B hasta el siguiente ciclo.
                    if is_obj:
                        state = 1; obs_r=r_val; obs_h=col_h[u]; obs_u=u

    # =========================================================================
    # 6. CLUSTERING (GRANULAR)
    # =========================================================================
    def run_clustering(self, points, P_per_point, u_all, v_all):
        """
        Ejecuta clustering DBSCAN sobre los puntos confirmados como obstáculos.
        """
        # 1. Filtrar Candidatos
        obs_pts, mask_obs = self.filter_cluster_candidates(points, P_per_point)
        if len(obs_pts) < 10: return [], None
        
        # 2. DBSCAN
        labels = self.run_dbscan(obs_pts)
        
        # 3. Procesar Clusters y Colorear
        clusters, cloud_msg = self.process_clusters(obs_pts, labels, mask_obs, len(points), u_all, v_all)
         # Extraer lista de clusters para el Hull
        cluster_list = [c_pts for c_pts, _ in clusters]
        
        return cluster_list, cloud_msg

    def filter_cluster_candidates(self, points, P):
        """Filtra puntos que superan el umbral de probabilidad de obstáculo."""
        mask = P > self.prob_threshold_obs
        return points[mask], mask

    def run_dbscan(self, points):
        """Ejecuta el algoritmo DBSCAN de sklearn."""
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(points)
        return clustering.labels_

    def process_clusters(self, obs_points, labels, mask_obs, total_points, u_all, v_all):
        """
        Procesa los resultados de DBSCAN, filtra por altura y genera la nube de puntos coloreada.
        También verifica sombras para cada cluster confirmado.
        """
        valid_clusters = []
        
        # Buffers de salida
        r_c = []
        g_c = []
        b_c = []
        pts_c = []
        
        unique = set(labels)
        for lbl in unique:
            if lbl == -1: continue
            mask_lbl = labels == lbl
            c_pts = obs_points[mask_lbl]
            
            # Filtrar similares a suelo (por altura media Z)
            # Si un cluster tiene una altura media muy baja (cerca o debajo del suelo estimado), se descarta como ruido de suelo.
            if c_pts[:, 2].mean() < self.ground_z_threshold: continue
            
            # Cluster Válido -> Asignar Color
            hue = (lbl * 77 % 360) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            valid_clusters.append((c_pts, lbl))
            
            pts_c.append(c_pts)
            r_c.extend([int(r*255)]*len(c_pts))
            g_c.extend([int(g*255)]*len(c_pts))
            b_c.extend([int(b*255)]*len(c_pts))
            
            # Actualizar Creencia para clusters confirmados (Lógica de sombra por cluster)
            self.check_cluster_shadow(c_pts, lbl, u_all, v_all, mask_obs, mask_lbl)
            
        if not pts_c: return [], None
        
        all_pts = np.vstack(pts_c)
        rgbs = (np.array(r_c, dtype=np.uint32) << 16) | (np.array(g_c, dtype=np.uint32) << 8) | np.array(b_c, dtype=np.uint32)
        
        msg = self.create_cloud(all_pts, rgbs.view(np.float32), 'rgb')
        return valid_clusters, msg

    def check_cluster_shadow(self, pts, label, u_all, v_all, mask_obs, mask_lbl_in_obs):
        """
        Verifica si un OBJECTO COMPLETO (Cluster) proyecta una sombra fuerte.
        
        DIFERENCIA CON `detect_geometric_shadows`:
        - `detect_geometric_shadows` (Paso 3, rápido): Mira columna a columna si hay píxeles vacíos tras un obstáculo. Es local.
        - `check_cluster_shadow` (Paso 6, lento): Mira el OBJETO ENTERO ya formado por DBSCAN.
          Calcula un "Score Global" de sombra. Si el objeto realmente es sólido, debería tener
          una gran área de vacío detrás de él.
        - Si el score es alto (>0.6), reforzamos a lo bestia (+2.0) la creencia de todos sus puntos.
        
        INTEGRACIÓN CON CONCAVE HULL:
        - Usamos el hull del frame anterior (self.prev_hull) para limitar la distancia
          máxima de la sombra (d_far). Si el rayo de sombra choca con la frontera
          del hull, NO miramos más allá (no tiene sentido: ahí no hay datos).
        - Generamos un polígono visual (Marker) mostrando la zona de sombra recortada.
        """
        # Se necesitan índices u,v para boost
        full_indices = np.where(mask_obs)[0][mask_lbl_in_obs]
        u_c = u_all[full_indices]
        v_c = v_all[full_indices]
        
        # Calcular ángulos extremos del cluster (para raycast contra hull)
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        r_obj = np.linalg.norm(pts[:, :2], axis=1)
        theta_min, theta_max = theta.min(), theta.max()
        r_max = r_obj.max()
        
        # Obtener d_far desde la intersección rayo-hull
        d_far_left = self.compute_ray_intersection(theta_min, max_dist=50.0)
        d_far_right = self.compute_ray_intersection(theta_max, max_dist=50.0)
        d_far = max(d_far_left, d_far_right)  # Usar el más lejano como límite
        
        # Calcular score limitado por d_far
        score = self.calculate_shadow_score(pts, d_far=d_far)
        if score > self.shadow_score_threshold:
            self.belief_map[u_c, v_c] += self.shadow_boost_val
            np.clip(self.belief_map, self.belief_clamp_min, self.belief_clamp_max, out=self.belief_map)
            self.add_shadow_text(pts, score)
            # Generar polígono visual de sombra recortado al hull
            self.compute_cluster_shadow_polygon(pts, theta_min, theta_max, r_max, d_far_left, d_far_right)

    def calculate_shadow_score(self, pts, d_far=None):
        """
        Calcula el 'Shadow Score' (0.0 a 1.0) usando GEOMETRÍA POLAR (Ángulos).
        
        ETAPA 2 (Aquí): Enfoque ANGULAR (más preciso para objetos irregulares).
        1. Obtenemos el ángulo mínimo y máximo (theta) que abarca el objeto desde el sensor.
        2. Lanzamos 'Rayos Virtuales' (steps) barriendo ese sector angular.
        3. Para cada rayo, miramos la columna de la Range Image correspondiente.
        4. Verificamos si DETRÁS de la distancia del objeto (max_r), los píxeles son VACÍO.
        
        INTEGRACIÓN CON HULL:
        - Si d_far está disponible (intersección con Concave Hull), solo miramos
          hasta esa distancia. Los píxeles más allá del hull no aportan información.
        - Esto evita falsos positivos en zonas donde simplemente no hay datos LiDAR.

                
        Diferencia con ETAPA 1 (`detect_geometric_shadows`):
        - La Etapa 1 usa Geometría de Triángulos (Altura/Distancia) en cada columna independiente.
        - La Etapa 2 usa Coherencia Angular: "Si esto es un coche, todo el ángulo detrás de él debe estar vacío".
        """
        if len(pts) < 5: return 0.0
        r = np.linalg.norm(pts[:,:2], axis=1)
        theta = np.arctan2(pts[:,1], pts[:,0])
        
        min_t, max_t = theta.min(), theta.max()
        max_r = r.max()
        
        # Límite superior: hull boundary o 60m (fallback)
        range_limit = d_far if d_far is not None else 60.0
        
        steps = 10 if (max_t - min_t) > 0.01 else 3
        angles = np.linspace(min_t, max_t, steps)
        
        shadow_px = 0
        total_px = 0
        
        # Pre-calcular rango esperado de suelo para cada fila de la mitad inferior.
        # Esto nos dice "a qué distancia DEBERÍA estar el suelo" en cada píxel.
        # Si un píxel es inf pero su suelo esperado está fuera del hull → lo ignoramos.
        u_lower = np.arange(self.H // 2, self.H)
        pitch_lower = np.radians((1.0 - u_lower / self.H) * 28.0 - 25.0)
        sin_p = np.sin(-pitch_lower)
        r_exp_lower = np.where(sin_p > 0.01, 1.73 / sin_p, 999.0)
        
        for ang in angles:
            v = int(0.5 * (ang/np.pi + 1.0) * self.W) % self.W
            col = self.current_range_image[:, v]
            lower = col[self.H//2:]
            
            # Clasificar cada píxel de la mitad inferior:
            is_behind = lower > (max_r + 0.5)            # Detrás del objeto (finitos)
            is_inf = np.isinf(lower)                      # Vacío total (sin retorno)
            
            # inf DENTRO del hull: su suelo esperado cae entre el objeto y el hull
            inf_in_hull = is_inf & (r_exp_lower > (max_r + 0.5)) & (r_exp_lower < range_limit)
            
            # Finitos dentro del hull: retorno real entre objeto y hull
            finite_in_hull = is_behind & (lower < range_limit)
            
            # Solo píxeles relevantes (dentro del hull)
            relevant = inf_in_hull | finite_in_hull
            n_relevant = np.sum(relevant)
            
            if n_relevant > 0:
                # inf dentro del hull = sombra confirmada
                # Finito dentro del hull = suelo detectado (dilución)
                shadow_px += np.sum(inf_in_hull)
                total_px += n_relevant
                
        return shadow_px / total_px if total_px > 0 else 0.0

    def add_shadow_text(self, pts, score):
        """Añade un marcador de texto sobre el objeto con su puntuación de sombra."""
        m = Marker()
        m.header.frame_id = "velodyne"
        m.ns = "shadow_scores"
        m.id = len(self.shadow_text_markers.markers)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.z = 0.5
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = f"CONFIRMED\nScore: {score:.2f}"
        
        c = np.mean(pts, axis=0)
        m.pose.position.x = float(c[0])
        m.pose.position.y = float(c[1])
        m.pose.position.z = float(c[2] + 1.0)
        m.pose.orientation.w = 1.0
        
        self.shadow_text_markers.markers.append(m)

    def compute_ray_intersection(self, angle, max_dist=50.0):
        """
        Lanza un rayo desde el origen en la dirección 'angle' y calcula dónde
        intersecta con la Concave Hull del frame anterior.
        
        GEOMETRÍA:
        - Rayo: P(t) = t * (cos(angle), sin(angle)),  t > 0
        - Para cada arista del hull (segmento A-B), resolvemos la intersección
          rayo-segmento usando el método de productos cruzados.
        - Retornamos la distancia al punto de intersección más cercano.
        - Si no hay hull (primer frame) o no hay intersección, retornamos max_dist.
        
        VERSIÓN VECTORIZADA: Procesa todas las aristas del hull a la vez con NumPy.
        ~10x más rápido que un for loop para hulls de 100-200 vértices.
        """
        if self.prev_hull is None or len(self.prev_hull) < 3:
            return max_dist
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        hull = self.prev_hull
        n = len(hull)
        
        # Vértices A (inicio) y B (fin) de todas las aristas, vectorizados
        hp1 = hull                        # (N, 2) - puntos A
        hp2 = np.roll(hull, -1, axis=0)   # (N, 2) - puntos B (siguiente vértice)
        
        # Vector de cada arista: E = B - A
        vx = hp2[:, 0] - hp1[:, 0]  # (N,)
        vy = hp2[:, 1] - hp1[:, 1]  # (N,)
        
        # Determinante: det = dx*vy - dy*vx
        denom = dx * vy - dy * vx  # (N,)
        
        # Filtrar aristas paralelas al rayo (det ≈ 0)
        valid = np.abs(denom) > 1e-10
        if not np.any(valid):
            return max_dist
        
        # Solo procesar aristas no paralelas
        p1x = hp1[valid, 0]
        p1y = hp1[valid, 1]
        ev_x = vx[valid]
        ev_y = vy[valid]
        d = denom[valid]
        
        # Parámetros de intersección:
        # t = distancia a lo largo del rayo (debe ser > 0)
        # s = posición en la arista (debe ser 0 <= s <= 1)
        t = (p1x * ev_y - p1y * ev_x) / d
        s = (p1x * dy - p1y * dx) / d
        
        # Intersecciones válidas: t > 0.1 y 0 <= s <= 1
        mask_hit = (t > 0.1) & (s >= 0.0) & (s <= 1.0)
        
        if np.any(mask_hit):
            return float(np.min(t[mask_hit]))
        
        return max_dist

    def compute_cluster_shadow_polygon(self, pts, theta_min, theta_max, r_max, d_far_left, d_far_right):
        """
        Genera una malla TRIANGLE_LIST que visualiza la zona de sombra de un cluster.
        
        En lugar de un simple trapecio, lanza rayos cada 0.5° dentro del ángulo
        del objeto y genera pares de puntos (d_near, d_far) para cada rayo.
        Cada par de rayos consecutivos genera 2 triángulos → malla curva rellena.
        
        La forma resultante sigue la silueta del objeto y se recorta al hull.
        
        Se publica como Marker.TRIANGLE_LIST en el topic /geometric_shadows.
        """
        z_vis = self.hull_vis_z
        step_rad = np.radians(0.5)  # Resolución angular: 0.5°
        
        # Cálculo polar del cluster
        r_pts = np.linalg.norm(pts[:, :2], axis=1)
        theta_pts = np.arctan2(pts[:, 1], pts[:, 0])
        
        # Generar ángulos de muestreo
        angles = np.arange(theta_min, theta_max + 1e-6, step_rad)
        if len(angles) < 2:
            angles = np.array([theta_min, theta_max])
        
        # Para cada ángulo, encontrar d_near (borde del objeto) y d_far (hull)
        fan_points = []
        tolerance_rad = step_rad
        
        for ang in angles:
            # d_near: distancia más lejana del objeto en esta dirección
            diff = np.abs(np.arctan2(np.sin(theta_pts - ang), np.cos(theta_pts - ang)))
            mask_beam = diff < tolerance_rad
            d_near = np.max(r_pts[mask_beam]) if np.any(mask_beam) else np.max(r_pts)
            
            # d_far: intersección con el hull en esta dirección
            d_far = self.compute_ray_intersection(ang)
            d_far = max(d_far, d_near + 0.1)  # Mínimo 10cm de sombra
            
            p_in = Point(x=float(d_near * np.cos(ang)), y=float(d_near * np.sin(ang)), z=float(z_vis))
            p_out = Point(x=float(d_far * np.cos(ang)), y=float(d_far * np.sin(ang)), z=float(z_vis))
            fan_points.append((p_in, p_out))
        
        if len(fan_points) < 2:
            return
        
        # Construir TRIANGLE_LIST a partir de pares consecutivos de rayos
        m = Marker()
        m.header.frame_id = "velodyne"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "cluster_shadows"
        m.id = len(self.shadow_text_markers.markers) + 100
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0; m.scale.y = 1.0; m.scale.z = 1.0
        m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.4)  # Azul semitransparente
        m.lifetime.sec = 0
        
        for i in range(len(fan_points) - 1):
            p1_in, p1_out = fan_points[i]
            p2_in, p2_out = fan_points[i + 1]
            
            # Triángulo 1: near_i → far_i → far_i+1
            m.points.append(p1_in)
            m.points.append(p1_out)
            m.points.append(p2_out)
            
            # Triángulo 2: near_i → far_i+1 → near_i+1
            m.points.append(p1_in)
            m.points.append(p2_out)
            m.points.append(p2_in)
        
        self.shadow_pub.publish(m)

    # =========================================================================
    # 7. HULL (SOLO CÓNCAVO)
    # =========================================================================
    def compute_concave_hull(self, points):
        """
        Calcula el Hull Cóncavo usando 'Frontier Sampling' y Triangulación de Delaunay + Alpha Shapes.
        No calcula Convex Hull.
        """
        if len(points) < 4: return
        
        # 1. Muestreo de Frontera (Frontier Sampling)
        frontier = self.sample_frontier(points)
        
        # 2. Añadir extremos de clusters
        frontier = self.add_cluster_extremes(frontier)
        
        # 3. Delaunay
        try:
            tri = Delaunay(frontier)
        except: return
        
        # 4. Filtrar triángulos (Alpha Shape Adaptive)
        edges = self.filter_triangles(tri, frontier)
        
        # 5. Extraer Borde (Boundary)
        boundary = self.extract_boundary(edges)
        if len(boundary) < 3: return
        
        # 6. Suavizar (Chaikin)
        smooth = smooth_chaikin(frontier[boundary], iterations=2, closed=True)
        
        # 7. Guardar y Visualizar
        self.publish_hull(smooth)
        self.points_2d = smooth
        
        # Guardar hull para el frame siguiente (usado por check_cluster_shadow)
        self.prev_hull = smooth.copy()
        
        # Reconstruir índices para lógica futura
        n = len(smooth)
        idx = np.arange(n)
        self.concave_hull_indices = np.stack((idx, np.roll(idx, -1)), axis=1)

    def sample_frontier(self, points):
        """
        Muestrea los puntos más lejanos en cada sector angular (Frontera de Visibilidad).
        Reduce dramáticamente el número de puntos para Delaunay.
        """
        xy = points[:, :2]
        r = np.linalg.norm(xy, axis=1)
        valid = r < self.max_range
        xy = xy[valid]
        r = r[valid]
        
        theta = np.arctan2(xy[:,1], xy[:,0])
        n_sectors = 2048
        sec_idx = ((theta + np.pi)/(2*np.pi)*n_sectors).astype(np.int32)
        sec_idx = np.clip(sec_idx, 0, n_sectors-1)
        
        # Max R por sector
        order = np.lexsort((-r, sec_idx))
        _, uniq = np.unique(sec_idx[order], return_index=True)
        return xy[order[uniq]]

    def add_cluster_extremes(self, frontier):
        """Añade los puntos extremos de los clusters detectados para asegurar su inclusión en el mapa."""
        if hasattr(self, 'detected_clusters'):
            extras = []
            for c in self.detected_clusters:
                xy = c[:, :2]
                extras.append([xy[:,0].min(), xy[:,1].min()])
                extras.append([xy[:,0].max(), xy[:,1].max()])
                extras.append([xy[:,0].min(), xy[:,1].max()])
                extras.append([xy[:,0].max(), xy[:,1].min()])
            if extras:
                return np.vstack([frontier, np.array(extras)])
        return frontier

    def filter_triangles(self, tri, pts):
        """Filtra triángulos de Delaunay basándose en el radio de su circunferencia circunscrita (Alpha Shape)."""
        coords = pts[tri.simplices]
        a = np.linalg.norm(coords[:,0]-coords[:,1], axis=1)
        b = np.linalg.norm(coords[:,1]-coords[:,2], axis=1)
        c = np.linalg.norm(coords[:,2]-coords[:,0], axis=1)
        s = (a+b+c)/2
        area = np.sqrt(np.maximum(0, s*(s-a)*(s-b)*(s-c)))
        
        r_circ = (a*b*c) / (4*area + 1e-6)
        
        # Alpha Adaptativo: Permite triángulos más grandes a mayor distancia
        means = np.linalg.norm(coords.mean(axis=1), axis=1)
        thresh = np.maximum(4.0, means * 0.2)
        
        valid = r_circ < thresh
        return tri.simplices[valid]

    def extract_boundary(self, simplices):
        """Extrae las aristas externas (frontera) de la malla de triángulos filtrada."""
        # 1. Conteo de Aristas (Count=1 -> Borde)
        edges = np.vstack([simplices[:,[0,1]], simplices[:,[1,2]], simplices[:,[2,0]]])
        edges.sort(axis=1) # Ordenar vértices para (u,v) == (v,u)
        
        # Vista estructurada para unique count rápido
        edges_view = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
        unique_edges_view, inverse, counts = np.unique(edges_view, return_inverse=True, return_counts=True)
        
        # Aristas de borde aparecen exactamente una vez
        boundary_edges_idx = np.where(counts == 1)[0]
        
        if len(boundary_edges_idx) < 3: return []
        
        # Recuperar índices originales
        _, unique_indices = np.unique(edges_view, return_index=True)
        boundary_indices_in_input = unique_indices[boundary_edges_idx]
        boundary_edges = edges[boundary_indices_in_input]
        
        # 2. Recorrido de Grafo (Ordenar aristas en polígono)
        adj = {}
        for u, v in boundary_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            
        start_node = boundary_edges[0,0]
        
        polygon_indices = [start_node]
        current = start_node
        visited = {start_node}
        
        while True:
            neighbors = adj.get(current, [])
            next_node = None
            for n in neighbors:
                if n == start_node and len(polygon_indices) > 2:
                    next_node = n; break # Lazo cerrado
                if n not in visited:
                    next_node = n; break # Continuar camino
            
            if next_node is None: break
            if next_node == start_node: break
            
            polygon_indices.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return np.array(polygon_indices)

    def publish_hull(self, points_2d):
        """Publica el Hull Cóncavo como un Marker LINE_STRIP en ROS."""
        m = Marker()
        m.header.frame_id = "velodyne"
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.15
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        
        for p in points_2d:
            val = Point()
            val.x = float(p[0])
            val.y = float(p[1])
            val.z = self.hull_vis_z
            m.points.append(val)
        # Cerrar loop
        m.points.append(m.points[0])
        
        self.hull_pub.publish(m)

    # =========================================================================
    # 8. VISUALIZACIÓN
    # =========================================================================
    def generate_visualizations(self, points, range_image, delta_r, P, ground, n_per_point, u, v):
        """Genera y almacena todos los mensajes PointCloud2 y Markers para su publicación."""
        
        # ========================================
        # CLOUD 1: PRE-FILTER (Delta R sin filtrar)
        # ========================================
        # Colores:
        #   Azul (0,0,255)    = Obstáculo (delta_r < threshold)
        #   Gris (128,128,128) = Suelo (delta_r ≈ 0)
        #   Naranja (255,165,0) = Depresión (0.2 < delta_r < 2.0)
        #   Cyan (0,255,255)   = Cielo/Lejano (delta_r >= 2.0)
        r_c = np.full(len(points), 128, dtype=np.uint32)
        g_c = np.full(len(points), 128, dtype=np.uint32)
        b_c = np.full(len(points), 128, dtype=np.uint32)
        
        mask_obs_raw = delta_r < self.threshold_obs
        mask_dep = (delta_r > 0.2) & (delta_r < 2.0)
        mask_sky = delta_r >= 2.0
        
        # Máscara combinada: obstáculo por threshold O por bins rechazados (paredes)
        mask_obs_combined = mask_obs_raw | self.rejected_mask
        
        # Azul = obstáculo
        r_c[mask_obs_combined] = 0; g_c[mask_obs_combined] = 0; b_c[mask_obs_combined] = 255
        
        # Depresión y Cielo no deben sobreescribir obstáculos
        mask_dep_clean = mask_dep & ~mask_obs_combined
        mask_sky_clean = mask_sky & ~mask_obs_combined
        
        # Naranja = depresión
        r_c[mask_dep_clean] = 255; g_c[mask_dep_clean] = 165; b_c[mask_dep_clean] = 0
        # Cyan = cielo/lejano
        r_c[mask_sky_clean] = 0; g_c[mask_sky_clean] = 255; b_c[mask_sky_clean] = 255
        
        rgb_pre = (r_c << 16) | (g_c << 8) | b_c
        self.delta_r_msg = self.create_cloud(points, rgb_pre.view(np.float32), 'rgb')
        
        # =========================================
        # CLOUD 2: POST-FILTER (Diff del filtro inter-ring)
        # =========================================
        # Muestra qué puntos cambió el filtro inter-ring:
        #   Rojo (255,0,0) = Eliminado por el filtro (era obstáculo, ya no lo es)
        #   Verde (0,255,0) = Añadido por el filtro (no era obstáculo, ahora sí)
        mask_obs_smooth = P > self.prob_threshold_obs  # Obstáculo post-filter
        
        r_f = r_c.copy()
        g_f = g_c.copy()
        b_f = b_c.copy()
        
        # Eliminados: eran obstáculo en raw pero NO después del filtro
        mask_removed = mask_obs_combined & ~mask_obs_smooth & ~self.rejected_mask
        # Añadidos: NO eran obstáculo en raw pero SÍ después del filtro
        mask_added = ~mask_obs_combined & mask_obs_smooth
        
        r_f[mask_removed] = 255; g_f[mask_removed] = 0; b_f[mask_removed] = 0
        r_f[mask_added] = 0; g_f[mask_added] = 255; b_f[mask_added] = 0
        
        rgb_post = (r_f << 16) | (g_f << 8) | b_f
        self.delta_r_filtered_msg = self.create_cloud(points, rgb_post.view(np.float32), 'rgb')
        
        # 3. Nube Bayes
        self.bayes_cloud_msg = self.encode_cloud_rgb(points, P, 'bayes')
        
        # 4. Nube Suelo
        self.ground_cloud_msg = self.create_cloud(ground, np.zeros(len(ground)), 'intensity')
        
        # 5. Imagen de Rango
        self.range_image_msg = self.cv_bridge.cv2_to_imgmsg(range_image, encoding="32FC1")
        self.range_image_msg.header.frame_id = "velodyne"
        
        # 6. Nube GT + Métricas de Precisión
        if self.has_gt:
            self.gt_cloud_msg = self.create_cloud(points, self.gt_semantic.astype(np.float32), 'intensity')
            
            # --- Métricas GT (SemanticKITTI) ---
            GROUND_IDS = [40, 44, 48, 49, 60, 72]
            OBSTACLE_IDS = [10, 11, 13, 15, 16, 18, 20,   # vehículos
                            30, 31, 32,                     # humanos
                            50, 51, 52,                     # estructuras
                            70, 71,                         # vegetación
                            80, 81, 99,                     # mobiliario
                            252, 253, 254, 255, 256, 257, 258, 259]  # en movimiento
            gt_is_obstacle = np.isin(self.gt_semantic, OBSTACLE_IDS)
            gt_is_ground = np.isin(self.gt_semantic, GROUND_IDS)
            labeled = ~np.isin(self.gt_semantic, [0, 1])  # Excluir unlabeled/outlier
            
            # Bayes Final (Temporal + Espacial)
            pred_mask = P > 0.5
            TP = int(np.sum(pred_mask & gt_is_obstacle & labeled))
            FP = int(np.sum(pred_mask & gt_is_ground & labeled))
            FN = int(np.sum(~pred_mask & gt_is_obstacle & labeled))
            TN = int(np.sum(~pred_mask & gt_is_ground & labeled))
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  GT Metrics [Frame {self.current_scan}]\n"
                f"{'='*60}\n"
                f"  {'Metric':<12} {'Value':>8}\n"
                f"  {'-'*20}\n"
                f"  {'TP':<12} {TP:>8}\n"
                f"  {'FP':<12} {FP:>8}\n"
                f"  {'FN':<12} {FN:>8}\n"
                f"  {'TN':<12} {TN:>8}\n"
                f"  {'-'*20}\n"
                f"  {'Precision':<12} {prec:>8.3f}\n"
                f"  {'Recall':<12} {rec:>8.3f}\n"
                f"  {'F1':<12} {f1:>8.3f}\n"
                f"{'='*60}"
            )
        
        # 7. Nube de Sombras de Objetos (Shadow Cloud)
        self.shadow_cloud_msg = self.generate_shadow_cloud(range_image)

    def generate_shadow_cloud(self, range_image):
        """
        Reconstruye puntos 3D en las zonas de sombra (boost > 0) para visualización.
        
        Cada celda (u, v) donde shadow_boost_map > 0 representa una zona donde
        un objeto confirmó su solidez al ocluir el láser. Reconstruimos esos
        puntos usando la geometría de la range image (pitch, yaw) y coloreamos
        por intensidad del boost (más boost = azul más fuerte).
        """
        if not hasattr(self, 'shadow_boost_map'):
            return None
        
        boost = self.shadow_boost_map
        mask = boost > 0.01
        
        if not np.any(mask):
            return None
        
        u_idx, v_idx = np.where(mask)
        boost_vals = boost[mask]
        
        # Reconstruir coordenadas 3D usando la geometría esférica
        # Usamos la range_image real para obtener la distancia
        # Si no hay rango válido, usamos la distancia esperada del suelo
        r_vals = range_image[u_idx, v_idx]
        
        # Para píxeles sin retorno (vacíos = sombra real), estimamos la distancia
        # usando la distancia del suelo esperado
        invalid = (r_vals <= 0.5) | (r_vals > 60.0)
        if hasattr(self, 'shadow_pitch_map'):
            pitch = self.shadow_pitch_map[u_idx, 0]
        else:
            pitch = np.radians((1.0 - u_idx / self.H) * 28.0 - 25.0)
        
        yaw = (2.0 * v_idx / float(self.W) - 1.0) * np.pi
        
        # Para píxeles vacíos, proyectar al suelo estimado (z = -1.73)
        sin_p = np.sin(-pitch)
        where_valid = sin_p > 0.01
        r_gnd = np.where(where_valid, 1.73 / sin_p, 15.0)
        r_vals[invalid] = np.clip(r_gnd[invalid], 1.0, 30.0)
        
        # Conversión esférica -> cartesiana
        rcos = r_vals * np.cos(pitch)
        x = rcos * np.cos(yaw)
        y = rcos * np.sin(yaw)
        z = r_vals * np.sin(pitch)
        
        shadow_pts = np.column_stack((x, y, z))
        
        # Colorear: Cian (sombra débil) -> Azul (sombra fuerte)
        norm_boost = np.clip(boost_vals / boost_vals.max(), 0.0, 1.0)
        r_c = np.zeros(len(shadow_pts), dtype=np.uint32)
        g_c = ((1.0 - norm_boost) * 200).astype(np.uint32)  # Cian a 0
        b_c = np.full(len(shadow_pts), 255, dtype=np.uint32)  # Siempre azul
        
        rgb = (r_c << 16) | (g_c << 8) | b_c
        return self.create_cloud(shadow_pts, rgb.view(np.float32), 'rgb')

    def encode_cloud_rgb(self, points, values, mode):
        """Codifica valores (Delta R o Probabilidad) a colores RGB en una PointCloud2."""
        r = np.zeros(len(points), dtype=np.uint32)
        g = np.zeros(len(points), dtype=np.uint32)
        b = np.zeros(len(points), dtype=np.uint32)
        
        if mode == 'delta':
            mask_obs = values < self.threshold_obs
            r[mask_obs]=0; g[mask_obs]=0; b[mask_obs]=255
            mask_gnd = (values >= -0.2) & (values <= 0.2)
            r[mask_gnd]=128; g[mask_gnd]=128; b[mask_gnd]=128
        elif mode == 'bayes':
            mask_obs = values > 0.8
            r[mask_obs]=0; g[mask_obs]=0; b[mask_obs]=255
            mask_gnd = values < 0.2
            r[mask_gnd]=0; g[mask_gnd]=200; b[mask_gnd]=0
            
        rgb = (r << 16) | (g << 8) | b
        return self.create_cloud(points, rgb.view(np.float32), 'rgb')

    def create_cloud(self, points, values, field_name):
        """Crea un mensaje PointCloud2 denso y optimizado."""
        msg = PointCloud2()
        msg.header.frame_id = "velodyne"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name=field_name, offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(points)
        msg.is_dense = True
        
        arr = np.zeros(len(points), dtype=[('x','f4'),('y','f4'),('z','f4'),(field_name,'f4')])
        arr['x'] = points[:,0]
        arr['y'] = points[:,1]
        arr['z'] = points[:,2]
        arr[field_name] = values
        
        msg.data = arr.tobytes()
        return msg

    def publish_visualization(self):
        """Publica todos los mensajes almacenados (Ejecutado por Timer)."""
        t = self.get_clock().now().to_msg()
        if hasattr(self, 'delta_r_msg'): 
            self.delta_r_msg.header.stamp = t
            self.delta_r_cloud_pub.publish(self.delta_r_msg)
        if hasattr(self, 'bayes_cloud_msg'):
            self.bayes_cloud_msg.header.stamp = t
            self.bayes_cloud_pub.publish(self.bayes_cloud_msg)
        if hasattr(self, 'ground_cloud_msg'):
            self.ground_cloud_msg.header.stamp = t
            self.ground_cloud_pub.publish(self.ground_cloud_msg)
        if hasattr(self, 'cluster_points_msg') and self.cluster_points_msg:
            self.cluster_points_msg.header.stamp = t
            self.cluster_points_pub.publish(self.cluster_points_msg)
        if hasattr(self, 'range_image_msg'):
            self.range_image_msg.header.stamp = t
            self.range_image_pub.publish(self.range_image_msg)
            
        if hasattr(self, 'shadow_text_markers'):
             for m in self.shadow_text_markers.markers:
                 m.header.stamp = t
             self.text_pub.publish(self.shadow_text_markers)
        
        # Publicar nube post-filter (diff del filtro inter-ring)
        if hasattr(self, 'delta_r_filtered_msg'):
            self.delta_r_filtered_msg.header.stamp = t
            self.delta_r_filtered_pub.publish(self.delta_r_filtered_msg)
        
        # Publicar nube de sombras de objetos
        if hasattr(self, 'shadow_cloud_msg') and self.shadow_cloud_msg is not None:
            self.shadow_cloud_msg.header.stamp = t
            self.shadow_cloud_pub.publish(self.shadow_cloud_msg)
        
        # Publicar nube de vacíos (obstáculos negativos)
        if hasattr(self, 'void_points') and len(self.void_points) > 0:
            void_msg = self.create_cloud(self.void_points, np.zeros(len(self.void_points)), 'intensity')
            void_msg.header.stamp = t
            self.pub_voids.publish(void_msg)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/insia/lidar_ws/data_odometry_velodyne/dataset')
    parser.add_argument('--scene', default='00')
    parser.add_argument('--scan_start', default='0')
    parser.add_argument('--scan_end', default='4')
    parser.add_argument('--scan', default=None)
    
    clean_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    p_args = parser.parse_args(clean_args)
    
    s_start = int(p_args.scan) if p_args.scan else int(p_args.scan_start)
    s_end = int(p_args.scan) if p_args.scan else int(p_args.scan_end)
    
    node = ObjectsDetector(p_args.data_path, p_args.scene, s_start, s_end)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
