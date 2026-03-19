#!/usr/bin/env python3
"""
Ejemplo de integración completa para TFG
Combina paso_1.py (actual) + extensiones (voids, negativos, T_var)

Uso:
    python3 paso_1_integrated_example.py --scan 0
"""

import numpy as np
import sys
import argparse
from pathlib import Path

# Importar clase base de paso_1.py
sys.path.append('/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea')
from paso_1 import GroundSegmentationPipeline

# Importar extensiones
from paso_1_extensions import (
    compute_local_variance,
    detect_voids,
    detect_negative_obstacles,
    compute_integrity_score
)


class ExtendedPipeline(GroundSegmentationPipeline):
    """
    Pipeline extendido que combina:
    - Segmentación de suelo (paso_1.py)
    - Detección de voids (extensions)
    - Detección de obstáculos negativos (extensions)
    - Capa de integridad T_var (ROBIO 2024)
    """

    def __init__(self):
        super().__init__()
        self.local_planes = {}  # Cache de planos locales para compute_local_variance

        # Parámetros configurables
        self.params = {
            'positive_threshold': 0.3,      # Delta-r para obstáculos positivos (m)
            'negative_threshold': -0.3,     # Delta-r para obstáculos negativos (m)
            'void_jump_threshold': 2.0,     # Salto de profundidad para voids (m)
            'variance_threshold': 0.1,      # Varianza máxima para voids (m²)
            'min_cluster_size': 10,         # Tamaño mínimo de cluster negativo
            'integrity_threshold': 0.5      # Score mínimo de integridad
        }

    def segment_ground_extended(self, points):
        """
        Versión extendida de segment_ground que cachea local_planes.
        """
        self.patchwork.estimateGround(points)
        ground_points = self.patchwork.getGround()

        centers = self.patchwork.getCenters()
        normals = self.patchwork.getNormals()

        # Filtrar planos verticales (paredes)
        local_planes, rejected_bins = self.filter_wall_planes(centers, normals)

        # NUEVO: Cachear local_planes para compute_local_variance
        self.local_planes = local_planes

        global_n, global_d = self.compute_global_plane(ground_points)
        n_per_point, d_per_point = self.assign_planes_to_points(
            points, local_planes, global_n, global_d
        )

        rejected_mask = self.create_rejected_mask(points, rejected_bins)

        return ground_points, n_per_point, d_per_point, rejected_mask

    # Añadir métodos de extensions como métodos de clase
    compute_local_variance = compute_local_variance
    detect_voids = detect_voids
    detect_negative_obstacles = detect_negative_obstacles
    compute_integrity_score = compute_integrity_score

    def process_frame_full(self, scan_file):
        """
        Pipeline completo: ground segmentation + delta_r + voids + negativos + integridad.

        Returns:
            results: Dict con todas las detecciones y métricas
        """
        # Cargar puntos
        points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]

        print(f"[INFO] Procesando {len(points)} puntos de {scan_file}")

        # Paso 1: Segmentación de suelo con filtrado de paredes
        print("[1/6] Segmentación de suelo con Patchwork++...")
        ground_points, n_per_point, d_per_point, rejected_mask = self.segment_ground_extended(points)
        print(f"      → Ground: {len(ground_points)} pts, Rechazados: {np.sum(rejected_mask)} pts")

        # Paso 2: Delta-r (anomalía de rango)
        print("[2/6] Calculando delta_r (rango esperado vs medido)...")
        r_expected = self.compute_expected_range(points, n_per_point, d_per_point)
        r_measured = np.linalg.norm(points, axis=1)
        delta_r = r_measured - r_expected
        print(f"      → Delta_r: mean={np.mean(delta_r):.3f}m, std={np.std(delta_r):.3f}m")

        # Paso 3: Varianza local (T_var)
        print("[3/6] Calculando varianza local por bin (T_var)...")
        variance = self.compute_local_variance(points, self.local_planes)
        print(f"      → Varianza: mean={np.mean(variance):.4f}m², max={np.max(variance):.4f}m²")

        # Paso 4: Obstáculos positivos
        print("[4/6] Detectando obstáculos positivos...")
        positive_mask = (delta_r > self.params['positive_threshold']) & ~rejected_mask
        n_positive = np.sum(positive_mask)
        print(f"      → Positivos: {n_positive} pts ({100*n_positive/len(points):.2f}%)")

        # Paso 5: Obstáculos negativos (baches)
        print("[5/6] Detectando obstáculos negativos (baches)...")
        negative_mask, negative_clusters = self.detect_negative_obstacles(
            points, delta_r, rejected_mask, variance,
            negative_threshold=self.params['negative_threshold'],
            min_cluster_size=self.params['min_cluster_size']
        )
        n_negative = np.sum(negative_mask)
        print(f"      → Negativos: {n_negative} pts en {len(negative_clusters)} clusters")

        # Paso 6: Voids (discontinuidades de profundidad)
        print("[6/6] Detectando voids (anomalías de visibilidad)...")
        void_mask, void_clusters = self.detect_voids(
            points, delta_r, rejected_mask, variance,
            void_threshold=self.params['void_jump_threshold'],
            var_threshold=self.params['variance_threshold']
        )
        n_voids = np.sum(void_mask)
        print(f"      → Voids: {n_voids} pts en {len(void_clusters)} clusters")

        # Paso 7: Score de integridad
        print("[EXTRA] Calculando score de integridad...")
        integrity = self.compute_integrity_score(points, delta_r, variance, rejected_mask)
        print(f"        → Integrity: mean={np.mean(integrity):.3f}, "
              f"low_conf={np.sum(integrity < self.params['integrity_threshold'])} pts")

        # Filtrar obstáculos positivos por integridad
        positive_mask_filtered = positive_mask & (integrity > self.params['integrity_threshold'])
        n_positive_filtered = np.sum(positive_mask_filtered)
        print(f"        → Positivos filtrados por integridad: {n_positive_filtered} pts "
              f"({n_positive - n_positive_filtered} descartados)")

        return {
            'points': points,
            'ground_points': ground_points,
            'delta_r': delta_r,
            'variance': variance,
            'integrity': integrity,
            'positive_mask': positive_mask,
            'positive_mask_filtered': positive_mask_filtered,
            'negative_mask': negative_mask,
            'negative_clusters': negative_clusters,
            'void_mask': void_mask,
            'void_clusters': void_clusters,
            'rejected_mask': rejected_mask
        }

    def compute_expected_range(self, points, n_per_point, d_per_point):
        """
        Calcula rango esperado basado en planos asignados (del paso_1.py original).
        """
        r_expected = np.zeros(len(points))

        for i in range(len(points)):
            n = n_per_point[i]
            d = d_per_point[i]

            if np.linalg.norm(n) < 1e-6:
                r_expected[i] = np.linalg.norm(points[i])
                continue

            # Proyección del rayo sobre el plano
            x, y, z = points[i]
            r_measured = np.sqrt(x**2 + y**2 + z**2)

            if r_measured < 1e-6:
                r_expected[i] = 0.0
                continue

            ray_dir = points[i] / r_measured
            denom = np.dot(n, ray_dir)

            if abs(denom) < 1e-6:
                r_expected[i] = r_measured
                continue

            t = -d / denom
            if t > 0:
                r_expected[i] = t
            else:
                r_expected[i] = r_measured

        return r_expected

    def save_results(self, results, output_dir):
        """
        Guarda resultados en formato compatible con visualización RViz.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        points = results['points']

        # Guardar nubes de puntos clasificadas
        np.save(output_dir / 'positive_obstacles.npy',
                points[results['positive_mask_filtered']])
        np.save(output_dir / 'negative_obstacles.npy',
                points[results['negative_mask']])
        np.save(output_dir / 'voids.npy',
                points[results['void_mask']])
        np.save(output_dir / 'rejected_walls.npy',
                points[results['rejected_mask']])

        # Guardar métricas por punto
        np.save(output_dir / 'delta_r.npy', results['delta_r'])
        np.save(output_dir / 'variance.npy', results['variance'])
        np.save(output_dir / 'integrity.npy', results['integrity'])

        # Guardar estadísticas
        stats = {
            'n_points': len(points),
            'n_ground': len(results['ground_points']),
            'n_positive': np.sum(results['positive_mask_filtered']),
            'n_negative': np.sum(results['negative_mask']),
            'n_voids': np.sum(results['void_mask']),
            'n_rejected': np.sum(results['rejected_mask']),
            'mean_delta_r': float(np.mean(results['delta_r'])),
            'std_delta_r': float(np.std(results['delta_r'])),
            'mean_variance': float(np.mean(results['variance'])),
            'mean_integrity': float(np.mean(results['integrity']))
        }

        import json
        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n[GUARDADO] Resultados en {output_dir}/")
        print(f"           - positive_obstacles.npy: {stats['n_positive']} pts")
        print(f"           - negative_obstacles.npy: {stats['n_negative']} pts")
        print(f"           - voids.npy: {stats['n_voids']} pts")
        print(f"           - stats.json: métricas globales")


def main():
    parser = argparse.ArgumentParser(description='Pipeline integrado TFG')
    parser.add_argument('--scan', type=int, default=0,
                        help='Número de scan a procesar (default: 0)')
    parser.add_argument('--output', type=str,
                        default='/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/results_integrated',
                        help='Directorio de salida')
    args = parser.parse_args()

    # Ruta al dataset
    data_dir = Path('/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/test_data/sequences/00/velodyne')
    scan_file = data_dir / f'{args.scan:06d}.bin'

    if not scan_file.exists():
        print(f"[ERROR] Archivo no encontrado: {scan_file}")
        sys.exit(1)

    # Crear pipeline
    print("="*70)
    print(" PIPELINE INTEGRADO TFG - Detección de Obstáculos Off-Road")
    print("="*70)
    print(f" Dataset: {scan_file}")
    print(f" Algoritmo: Patchwork++ + Delta-r + Voids + Negativos + T_var")
    print("="*70 + "\n")

    pipeline = ExtendedPipeline()

    # Procesar frame
    import time
    t_start = time.time()
    results = pipeline.process_frame_full(scan_file)
    t_elapsed = (time.time() - t_start) * 1000

    print(f"\n{'='*70}")
    print(f" TIEMPO TOTAL: {t_elapsed:.1f} ms")
    print(f"{'='*70}\n")

    # Guardar resultados
    pipeline.save_results(results, args.output)

    print("\n[OK] Pipeline completado exitosamente.")
    print(f"[INFO] Para visualizar en RViz, ejecuta:")
    print(f"       python3 visualize_integrated_results.py --input {args.output}")


if __name__ == '__main__':
    main()
