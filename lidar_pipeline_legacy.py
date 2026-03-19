#!/usr/bin/env python3
"""
Código legacy extraído de lidar_pipeline_suite.py

Contiene métodos que fueron reemplazados por la implementación per-point:
- compute_delta_r_on_range_image(): Delta-r sobre range image (reemplazado por compute_delta_r per-point)
- delta_r_to_binary_probability(): Conversión binaria (nunca usado)
- project_to_range_image(): Proyección a range image 2D (solo usado por stage3_complete)
- update_belief_map(): Bayesian filter sobre range image 2D (solo usado por stage3_complete)
- stage3_complete(): Stage 3 basado en range image (reemplazado por stage3_per_point)

El pipeline activo usa stage3_per_point() que trabaja directamente con puntos 3D,
evitando la compresión 20:1 que causa pérdida de información en la proyección a range image.

Estos métodos se mantienen aquí como referencia histórica.
"""

import numpy as np
from typing import Dict, Optional


def compute_delta_r_on_range_image(self, range_image, local_planes_dict, u, v, valid_mask, points):
    """
    Calcula delta_r y likelihood DIRECTAMENTE sobre range_image.
    REEMPLAZADO por compute_delta_r() per-point en stage2_complete().
    """
    H, W = range_image.shape
    delta_r_image = np.zeros((H, W), dtype=np.float32)
    likelihood_image = np.full((H, W), self.config.l0, dtype=np.float32)

    valid_pixels = range_image > 0
    if not np.any(valid_pixels):
        return {'delta_r_image': delta_r_image, 'likelihood_image': likelihood_image}

    u_pixels, v_pixels = np.where(valid_pixels)
    pixel_to_points = {}
    valid_idx = np.where(valid_mask)[0]

    for idx in valid_idx:
        pixel_key = (u[idx], v[idx])
        point = points[idx]
        r_measured = np.sqrt(np.sum(point**2))

        zone_idx, ring_idx, sector_idx = self.get_czm_bin(
            np.array([point[0]]), np.array([point[1]])
        )
        bin_id = (int(zone_idx[0]), int(ring_idx[0]), int(sector_idx[0]))

        if bin_id not in local_planes_dict:
            continue

        n, d = local_planes_dict[bin_id]
        ray_dir = point / r_measured
        dot_prod = np.dot(ray_dir, n)

        if abs(dot_prod) < 1e-6:
            continue

        r_expected = -d / dot_prod
        delta_r = r_measured - r_expected

        if delta_r < self.config.threshold_obs:
            likelihood = 2.0
        elif delta_r > self.config.threshold_void:
            likelihood = 1.5
        else:
            likelihood = -2.5

        if pixel_key not in pixel_to_points:
            pixel_to_points[pixel_key] = []
        pixel_to_points[pixel_key].append((idx, r_measured, delta_r, likelihood))

    for pixel_key, points_list in pixel_to_points.items():
        ui, vi = pixel_key
        max_likelihood_idx = np.argmax([p[3] for p in points_list])
        best_point = points_list[max_likelihood_idx]
        _, r_measured, delta_r, likelihood = best_point
        delta_r_image[ui, vi] = delta_r
        likelihood_image[ui, vi] = likelihood

    return {'delta_r_image': delta_r_image, 'likelihood_image': likelihood_image}


def delta_r_to_binary_probability(self, delta_r):
    """
    Convierte delta_r a PROBABILIDAD BINARIA.
    NUNCA USADO por el pipeline activo.
    """
    threshold_obs = self.config.threshold_obs
    probability = (delta_r < threshold_obs).astype(np.float32)
    return probability


def project_to_range_image(self, points, delta_r=None, likelihood=None, use_binary_probability=False):
    """
    Proyecta puntos 3D a range image 2D (H x W).
    SOLO USADO por stage3_complete() (código muerto).
    """
    N = len(points)
    H = self.config.range_image_height
    W = self.config.range_image_width

    r = np.sqrt(np.sum(points**2, axis=1))
    pitch = np.arcsin(np.clip(points[:, 2] / r, -1.0, 1.0))
    yaw = np.arctan2(points[:, 1], points[:, 0])

    fov_total = (self.config.fov_up - self.config.fov_down) * np.pi / 180.0
    fov_down_rad = self.config.fov_down * np.pi / 180.0

    proj_y = (pitch - fov_down_rad) / fov_total
    proj_y = 1.0 - proj_y
    u = np.floor(proj_y * H).astype(np.int32)
    u = np.clip(u, 0, H - 1)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    v = np.floor(proj_x * W).astype(np.int32)
    v = np.clip(v, 0, W - 1)

    valid_mask = (r > self.config.min_range) & (r < self.config.max_range)
    range_image = np.zeros((H, W), dtype=np.float32)
    likelihood_image = None

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]

        if likelihood is not None:
            likelihood_to_project = likelihood.copy()
            if use_binary_probability:
                likelihood_to_project = (likelihood > 0).astype(np.float32)

            likelihood_image = np.full((H, W), 0.0 if use_binary_probability else self.config.l0, dtype=np.float32)
            pixel_to_points = {}
            for i in valid_idx:
                pixel_key = (u[i], v[i])
                if pixel_key not in pixel_to_points:
                    pixel_to_points[pixel_key] = []
                pixel_to_points[pixel_key].append((i, r[i], likelihood_to_project[i]))

            for (ui, vi), points_list in pixel_to_points.items():
                max_likelihood = max([p[2] for p in points_list])
                epsilon = 0.01 if use_binary_probability else 0.1
                max_likelihood_points = [p for p in points_list if p[2] >= max_likelihood - epsilon]
                closest_of_max = min(max_likelihood_points, key=lambda p: p[1])
                _, r_best, likelihood_best = closest_of_max
                range_image[ui, vi] = r_best
                likelihood_image[ui, vi] = likelihood_best
        else:
            order = np.argsort(r[valid_idx])[::-1]
            u_sorted = u[valid_idx][order]
            v_sorted = v[valid_idx][order]
            r_sorted = r[valid_idx][order]
            range_image[u_sorted, v_sorted] = r_sorted

    return {
        'range_image': range_image,
        'likelihood_image': likelihood_image,
        'u': u, 'v': v,
        'valid_mask': valid_mask
    }


def update_belief_map(self, likelihood, range_proj, delta_pose=None):
    """
    Stage 3 Bayesian Temporal Filter sobre range image 2D.
    REEMPLAZADO por stage3_per_point() que trabaja directamente en 3D.
    """
    import time
    t_start = time.time()

    H = self.config.range_image_height
    W = self.config.range_image_width
    range_image = range_proj['range_image']

    if delta_pose is not None and hasattr(self, 'belief_map_prev'):
        belief_map_warped = self.belief_map_prev.copy()
        range_image_prev_warped = self.range_image_prev.copy()
    else:
        belief_map_warped = np.full((H, W), self.config.l0, dtype=np.float32)
        range_image_prev_warped = np.zeros((H, W), dtype=np.float32)

    depth_change = np.abs(range_image - range_image_prev_warped)
    reset_mask = depth_change > self.config.depth_jump_threshold
    belief_map_warped[reset_mask] = self.config.l0

    likelihood_image = range_proj.get('likelihood_image')
    if likelihood_image is None:
        raise ValueError("likelihood_image no encontrado en range_proj")

    valid_pixels = range_image > 0
    likelihood_min = np.min(likelihood_image[valid_pixels]) if np.any(valid_pixels) else 0.0
    likelihood_max_val = np.max(likelihood_image[valid_pixels]) if np.any(valid_pixels) else 0.0

    is_probability = (likelihood_min >= 0.0) and (likelihood_max_val <= 1.0)

    if is_probability:
        prob_clamped = np.clip(likelihood_image, 1e-6, 1.0 - 1e-6)
        likelihood_log_odds = np.log(prob_clamped / (1.0 - prob_clamped))
    else:
        prob_from_likelihood = 1.0 / (1.0 + np.exp(-likelihood_image))
        prob_clamped = np.clip(prob_from_likelihood, 1e-6, 1.0 - 1e-6)
        likelihood_log_odds = np.log(prob_clamped / (1.0 - prob_clamped))

    belief_map = likelihood_log_odds + belief_map_warped - 0.0
    belief_map = np.clip(belief_map, self.config.belief_clamp_min, self.config.belief_clamp_max)

    belief_threshold_obs = np.log(self.config.prob_threshold_obs / (1 - self.config.prob_threshold_obs))
    obs_belief_mask = (belief_map > belief_threshold_obs) & (range_image > 0)

    self.belief_map_prev = belief_map.copy()
    self.range_image_prev = range_image.copy()

    t_end = time.time()
    return {
        'belief_map': belief_map,
        'obs_belief_mask': obs_belief_mask,
        'belief_threshold_obs': belief_threshold_obs,
        'timing_ms': (t_end - t_start) * 1000.0
    }


def stage3_complete(self, points, delta_pose=None, use_binary_probability=True):
    """
    Stage 3 completo basado en range image.
    REEMPLAZADO por stage3_per_point() que evita la compresión 20:1.
    """
    self.points_current = points
    stage2_result = self.stage2_complete(points)

    likelihood_to_project = stage2_result['likelihood']
    if use_binary_probability:
        likelihood_to_project = (stage2_result['likelihood'] > 0).astype(np.float32)

    range_proj = self.project_to_range_image(
        points=points,
        likelihood=likelihood_to_project,
        use_binary_probability=False
    )

    stage3_result = self.update_belief_map(
        likelihood=stage2_result['likelihood'],
        range_proj=range_proj,
        delta_pose=delta_pose
    )

    return {
        **stage2_result,
        **stage3_result,
        'range_proj': range_proj,
        'timing_total_ms': stage2_result['timing_total_ms'] + stage3_result['timing_ms']
    }
