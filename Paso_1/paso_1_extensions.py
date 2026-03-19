"""
Extensiones para paso_1.py - Detección de voids y obstáculos negativos
Inspirado en metodología ROBIO 2024 (T_height + T_var)

Autor: Feedback para TFG LiDAR off-road
Fecha: 2026-03-02
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_local_variance(self, points, local_planes):
    """
    Calcula varianza local de altura por bin (T_var de ROBIO 2024).

    Alta varianza indica:
    - Terreno irregular (válido en off-road)
    - Transición ground/obstacle (válido para detección)
    - Void edge (crítico para detección de vacíos)

    Args:
        points: Nube de puntos (N, 3)
        local_planes: Dict {bin_id: (normal, d)}

    Returns:
        variance_per_point: Array (N,) con varianza del bin correspondiente
    """
    z_idx, r_idx, s_idx = self.get_czm_bin_vectorized(points[:, 0], points[:, 1])
    variance_per_point = np.zeros(len(points))

    for bin_id, (n, d) in local_planes.items():
        z_bin, r_bin, s_bin = bin_id
        bin_mask = (z_idx == z_bin) & (r_idx == r_bin) & (s_idx == s_bin)

        if np.sum(bin_mask) < 3:  # Mínimo 3 puntos para calcular varianza
            continue

        # Distancia firmada de cada punto al plano del bin
        bin_points = points[bin_mask]
        signed_distances = bin_points @ n + d

        # Varianza de las distancias al plano
        variance = np.var(signed_distances)
        variance_per_point[bin_mask] = variance

    return variance_per_point


def detect_voids(self, points, delta_r, rejected_mask, variance_per_point,
                 void_threshold=2.0, var_threshold=0.1):
    """
    Detecta voids (anomalías de visibilidad) mediante gradientes de profundidad.

    Voids aparecen como discontinuidades en rango donde:
    1. Salto de profundidad > threshold (ej: 2m)
    2. Baja varianza local (no es terreno irregular)
    3. No está en bin de pared rechazado

    Args:
        points: Nube de puntos (N, 3)
        delta_r: Anomalía de rango (N,) = r_measured - r_expected
        rejected_mask: Máscara de bins de paredes (N,)
        variance_per_point: Varianza local por punto (N,)
        void_threshold: Umbral de salto de profundidad (metros)
        var_threshold: Umbral de varianza (metros^2)

    Returns:
        void_mask: Máscara booleana de puntos en void edges
        void_clusters: Lista de clusters de voids detectados
    """
    ranges = np.linalg.norm(points[:, :2], axis=1)  # Distancia radial

    # Construir KDTree para búsqueda de vecinos en imagen de rango
    # (aproximación: usar vecinos 3D cercanos)
    tree = cKDTree(points)

    void_mask = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if rejected_mask[i]:  # Ignorar puntos en bins de paredes
            continue

        # Buscar 8 vecinos más cercanos (equivalente a 8-conectividad en imagen)
        dists, indices = tree.query(points[i], k=9)  # k=9 incluye el punto mismo
        neighbors = indices[1:]  # Excluir el punto mismo

        # Calcular saltos de profundidad respecto a vecinos
        range_jumps = np.abs(ranges[neighbors] - ranges[i])
        max_jump = np.max(range_jumps)

        # Criterios para void edge:
        # 1. Salto de profundidad grande
        # 2. Baja varianza local (no es irregularidad del terreno)
        if max_jump > void_threshold and variance_per_point[i] < var_threshold:
            void_mask[i] = True

    # Clustering de voids usando DBSCAN (similar a tu pipeline)
    from sklearn.cluster import DBSCAN

    void_points = points[void_mask]
    if len(void_points) > 0:
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(void_points)
        void_clusters = []

        for label in set(clustering.labels_):
            if label == -1:  # Ruido
                continue
            cluster_mask = clustering.labels_ == label
            void_clusters.append(void_points[cluster_mask])
    else:
        void_clusters = []

    return void_mask, void_clusters


def detect_negative_obstacles(self, points, delta_r, rejected_mask, variance_per_point,
                               negative_threshold=-0.3, min_cluster_size=10):
    """
    Detecta obstáculos negativos (baches, hundimientos) mediante delta_r < 0.

    Obstáculo negativo válido:
    1. delta_r < negative_threshold (punto por debajo del plano esperado)
    2. Cluster coherente (min_cluster_size puntos)
    3. No está en bin de pared rechazado
    4. Varianza moderada (diferencia de ruido puntual)

    Args:
        points: Nube de puntos (N, 3)
        delta_r: Anomalía de rango (N,) = r_measured - r_expected
        rejected_mask: Máscara de bins de paredes (N,)
        variance_per_point: Varianza local por punto (N,)
        negative_threshold: Umbral para clasificar como negativo (metros, negativo)
        min_cluster_size: Tamaño mínimo de cluster para considerar válido

    Returns:
        negative_mask: Máscara de obstáculos negativos válidos
        negative_clusters: Lista de clusters de obstáculos negativos
    """
    # Candidatos: delta_r negativo, no en paredes, varianza razonable
    negative_candidates = (delta_r < negative_threshold) & ~rejected_mask & (variance_per_point < 0.5)

    if np.sum(negative_candidates) == 0:
        return np.zeros(len(points), dtype=bool), []

    # Clustering con DBSCAN
    from sklearn.cluster import DBSCAN

    candidate_points = points[negative_candidates]
    clustering = DBSCAN(eps=0.4, min_samples=min_cluster_size).fit(candidate_points)

    negative_mask = np.zeros(len(points), dtype=bool)
    negative_clusters = []

    # Mapear clusters de vuelta a índices originales
    candidate_indices = np.where(negative_candidates)[0]

    for label in set(clustering.labels_):
        if label == -1:  # Ruido
            continue

        cluster_mask_local = clustering.labels_ == label
        cluster_indices = candidate_indices[cluster_mask_local]

        # Validar cluster: profundidad promedio suficiente
        cluster_delta_r = delta_r[cluster_indices]
        if np.mean(cluster_delta_r) < negative_threshold:  # Cluster coherente
            negative_mask[cluster_indices] = True
            negative_clusters.append(points[cluster_indices])

    return negative_mask, negative_clusters


def compute_integrity_score(self, points, delta_r, variance_per_point, rejected_mask):
    """
    Calcula score de integridad por punto (inspirado en ROBIO 2024).

    Score alto (cerca de 1.0) = confianza alta en clasificación
    Score bajo (cerca de 0.0) = zona sospechosa (requiere validación adicional)

    Factores:
    1. Varianza baja -> alta confianza
    2. Delta_r moderado -> alta confianza (terreno plano esperado)
    3. No en bin rechazado -> alta confianza

    Args:
        points: Nube de puntos (N, 3)
        delta_r: Anomalía de rango (N,)
        variance_per_point: Varianza local (N,)
        rejected_mask: Máscara de bins de paredes (N,)

    Returns:
        integrity_score: Array (N,) con score [0, 1]
    """
    # Normalizar varianza a [0, 1] (var > 1.0 = baja confianza)
    var_score = np.clip(1.0 - variance_per_point, 0.0, 1.0)

    # Normalizar delta_r a [0, 1] (|delta_r| > 1.0m = baja confianza)
    delta_score = np.clip(1.0 - np.abs(delta_r) / 1.0, 0.0, 1.0)

    # Penalizar bins rechazados
    rejection_score = (~rejected_mask).astype(float)

    # Score combinado (producto para que todos los factores sean necesarios)
    integrity_score = var_score * delta_score * rejection_score

    return integrity_score


# ============================================================================
# FUNCIÓN PRINCIPAL PARA INTEGRAR EN paso_1.py
# ============================================================================

def process_frame_extended(self, points):
    """
    Pipeline completo extendido para TFG: obstáculos positivos, negativos y voids.

    Esta función REEMPLAZA o EXTIENDE el flujo principal de paso_1.py.

    Returns:
        results: Dict con todas las detecciones
            - 'ground_points': Puntos clasificados como suelo
            - 'delta_r': Anomalía de rango por punto
            - 'positive_obstacles': Máscara de obstáculos positivos
            - 'negative_obstacles': Máscara de obstáculos negativos
            - 'voids': Máscara de void edges
            - 'integrity_score': Score de confianza por punto
            - 'rejected_mask': Bins de paredes rechazados
    """
    # Paso 1: Segmentación de suelo con filtrado de paredes
    ground_points, n_per_point, d_per_point, rejected_mask = self.segment_ground(points)

    # Paso 2: Cálculo de delta_r (rango esperado vs medido)
    r_expected = self.compute_expected_range(points, n_per_point, d_per_point)
    r_measured = np.linalg.norm(points, axis=1)
    delta_r = r_measured - r_expected

    # Paso 3: Varianza local (T_var de ROBIO 2024)
    variance_per_point = self.compute_local_variance(points, self.get_local_planes())

    # Paso 4: Detección de obstáculos positivos (delta_r > threshold)
    positive_threshold = 0.3  # metros (configurable)
    positive_mask = (delta_r > positive_threshold) & ~rejected_mask

    # Paso 5: Detección de obstáculos negativos (baches)
    negative_mask, negative_clusters = self.detect_negative_obstacles(
        points, delta_r, rejected_mask, variance_per_point
    )

    # Paso 6: Detección de voids (discontinuidades de profundidad)
    void_mask, void_clusters = self.detect_voids(
        points, delta_r, rejected_mask, variance_per_point
    )

    # Paso 7: Score de integridad
    integrity_score = self.compute_integrity_score(
        points, delta_r, variance_per_point, rejected_mask
    )

    return {
        'ground_points': ground_points,
        'delta_r': delta_r,
        'positive_obstacles': positive_mask,
        'negative_obstacles': negative_mask,
        'negative_clusters': negative_clusters,
        'voids': void_mask,
        'void_clusters': void_clusters,
        'integrity_score': integrity_score,
        'rejected_mask': rejected_mask,
        'variance': variance_per_point
    }
