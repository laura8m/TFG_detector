#!/usr/bin/env python3
"""
Script de debug para analizar la detección de paredes.
Muestra estadísticas de los planos locales y su clasificación.
"""

import numpy as np
from pathlib import Path
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

def analyze_wall_detection(scan_id=0, sequence='00'):
    """Analiza la detección de paredes en un scan"""

    # Cargar scan
    scan_path = Path(f'/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/{sequence}/{sequence}/velodyne/{scan_id:06d}.bin')
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]

    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE DETECCIÓN DE PAREDES - Scan {scan_id:06d}")
    print(f"{'='*80}\n")
    print(f"Total de puntos: {len(points)}")

    # Configuración con wall rejection activado
    config = PipelineConfig(
        enable_hybrid_wall_rejection=True,
        enable_hcd=False,
        verbose=False
    )

    # Ejecutar pipeline
    pipeline = LidarPipelineSuite(config, data_path=str(scan_path))
    result = pipeline.stage1_complete(points)

    # Analizar planos locales
    local_planes = pipeline.local_planes

    print(f"\nPlanos locales detectados: {len(local_planes)}")

    # Clasificar planos por verticalidad
    horizontal = []  # nz > 0.9 (muy horizontal)
    inclined = []    # 0.7 < nz <= 0.9 (inclinado)
    steep = []       # 0.5 < nz <= 0.7 (muy inclinado)
    vertical = []    # nz <= 0.5 (casi vertical)

    for bin_id, (n, d) in local_planes.items():
        nz = abs(n[2])

        if nz > 0.9:
            horizontal.append((bin_id, n, d, nz))
        elif nz > 0.7:
            inclined.append((bin_id, n, d, nz))
        elif nz > 0.5:
            steep.append((bin_id, n, d, nz))
        else:
            vertical.append((bin_id, n, d, nz))

    print(f"\nClasificación por inclinación:")
    print(f"  Horizontal (nz > 0.9):     {len(horizontal):4d} planos ({len(horizontal)/len(local_planes)*100:.1f}%)")
    print(f"  Inclinado (0.7 < nz ≤ 0.9): {len(inclined):4d} planos ({len(inclined)/len(local_planes)*100:.1f}%)")
    print(f"  Muy inclinado (0.5 < nz ≤ 0.7): {len(steep):4d} planos ({len(steep)/len(local_planes)*100:.1f}%)")
    print(f"  Vertical (nz ≤ 0.5):       {len(vertical):4d} planos ({len(vertical)/len(local_planes)*100:.1f}%)")

    # Analizar paredes rechazadas
    walls_rejected = result['rejected_walls']
    print(f"\n{'='*80}")
    print(f"RESULTADO:")
    print(f"{'='*80}")
    print(f"Ground points:  {len(result['ground_indices']):6d}")
    print(f"Paredes rechazadas: {len(walls_rejected):6d}")

    if len(walls_rejected) == 0:
        print("\n⚠️  NO SE DETECTARON PAREDES")
        print("\nPosibles razones:")
        print("  1. El threshold de nz=0.7 es demasiado estricto")
        print("  2. No hay estructuras verticales verdaderas en este scan")
        print("  3. El filtro geométrico local (delta_z > 0.3m) es muy estricto")
    else:
        print(f"\n✓ Se detectaron {len(walls_rejected)} puntos como paredes")

    # Mostrar ejemplos de planos verticales/inclinados
    if len(steep) > 0 or len(vertical) > 0:
        print(f"\n{'='*80}")
        print("PLANOS CANDIDATOS A PARED (no rechazados por filtro geométrico):")
        print(f"{'='*80}")

        candidates = steep + vertical
        for i, (bin_id, n, d, nz) in enumerate(candidates[:10]):  # Primeros 10
            angle = np.degrees(np.arccos(nz))
            print(f"  Bin {bin_id}: nz={nz:.3f}, ángulo={angle:.1f}°, normal={n}")

    # Sugerencias
    print(f"\n{'='*80}")
    print("SUGERENCIAS:")
    print(f"{'='*80}")

    if len(vertical) > 0:
        print(f"✓ Hay {len(vertical)} planos casi verticales (nz ≤ 0.5)")
        print("  → Estos están siendo considerados candidatos a pared")

    if len(steep) > 0:
        print(f"⚠️  Hay {len(steep)} planos muy inclinados (0.5 < nz ≤ 0.7)")
        print("  → Estos NO están siendo considerados (threshold actual: nz < 0.7)")
        print("  → SUGERENCIA: Reducir wall_rejection_slope a 0.5 o 0.6")

    if len(walls_rejected) == 0 and (len(steep) > 0 or len(vertical) > 0):
        print("\n⚠️  PROBLEMA: Hay planos candidatos pero no se rechazan puntos")
        print("  → El filtro geométrico local (delta_z) puede ser demasiado estricto")
        print("  → SUGERENCIA: Reducir wall_height_diff_threshold de 0.3m a 0.15m")

    print(f"\n{'='*80}\n")

    return {
        'total_planes': len(local_planes),
        'horizontal': len(horizontal),
        'inclined': len(inclined),
        'steep': len(steep),
        'vertical': len(vertical),
        'walls_rejected': len(walls_rejected)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debug wall detection')
    parser.add_argument('--scan', type=int, default=0, help='Scan ID')
    parser.add_argument('--sequence', type=str, default='00', help='KITTI sequence (00, 04, etc.)')
    args = parser.parse_args()

    analyze_wall_detection(args.scan, args.sequence)
