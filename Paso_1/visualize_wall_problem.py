#!/usr/bin/env python3
"""
Visualización clara del problema: Patchwork++ clasifica trozos de paredes como suelo.
Muestra los 11 segmentos verticales detectados (ΔZ > 0.5m) que fueron clasificados como 'suelo'.
"""

import numpy as np
import sys
from pathlib import Path

# Añadir Patchwork++ al path

def main():
    import pypatchworkpp

    # Cargar datos
    bin_file = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/data/000000.bin")
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]

    # Ejecutar Patchwork++ (configuración por defecto)
    params = pypatchworkpp.Parameters()
    params.verbose = False
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
    PatchworkPLUSPLUS.estimateGround(points)

    ground = PatchworkPLUSPLUS.getGround()
    nonground = PatchworkPLUSPLUS.getNonground()

    print("="*80)
    print("VISUALIZACIÓN DEL PROBLEMA: Paredes clasificadas como suelo")
    print("="*80)
    print()
    print(f"Total puntos:       {len(points):>8,}")
    print(f"Clasificados SUELO: {len(ground):>8,} ({100*len(ground)/len(points):>5.1f}%)")
    print(f"Clasificados NO-SUELO: {len(nonground):>8,} ({100*len(nonground)/len(points):>5.1f}%)")
    print()

    # Analizar segmentos verticales en puntos clasificados como "suelo"
    r = np.sqrt(ground[:, 0]**2 + ground[:, 1]**2)
    theta = np.arctan2(ground[:, 1], ground[:, 0])
    z = ground[:, 2]

    # Bins cilíndricos (1m radial, 5° angular)
    r_bins = np.arange(0, 80, 1.0)
    theta_bins = np.arange(-np.pi, np.pi, np.radians(5))

    r_idx = np.digitize(r, r_bins)
    theta_idx = np.digitize(theta, theta_bins)

    bin_keys = list(zip(r_idx, theta_idx))
    unique_bins = set(bin_keys)

    # Encontrar segmentos verticales (ΔZ > 0.5m)
    vertical_segments = []

    for r_id, theta_id in unique_bins:
        mask = (r_idx == r_id) & (theta_idx == theta_id)
        bin_points = ground[mask]

        if len(bin_points) < 5:
            continue

        z_min = np.min(bin_points[:, 2])
        z_max = np.max(bin_points[:, 2])
        delta_z = z_max - z_min

        if delta_z > 0.5:  # Threshold para pared
            r_center = r_bins[r_id-1] if r_id > 0 else 0
            theta_center = theta_bins[theta_id-1] if theta_id > 0 else 0

            vertical_segments.append({
                'r': r_center,
                'theta': theta_center,
                'n_points': len(bin_points),
                'z_min': z_min,
                'z_max': z_max,
                'delta_z': delta_z,
                'points': bin_points,
                'bin_id': (r_id, theta_id)
            })

    vertical_segments.sort(key=lambda x: x['delta_z'], reverse=True)

    print("="*80)
    print(f"⚠️  SEGMENTOS VERTICALES ENCONTRADOS: {len(vertical_segments)}")
    print("="*80)
    print()

    if len(vertical_segments) == 0:
        print("✅ No se encontraron segmentos verticales problemáticos.")
        return

    print("Estos segmentos tienen variación de altura > 0.5m → SON PAREDES, no suelo")
    print()

    # Mostrar detalles de cada segmento
    for i, seg in enumerate(vertical_segments, 1):
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"SEGMENTO #{i}: {'PARED VERTICAL' if seg['delta_z'] > 1.0 else 'PARED/RAMPA'}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        angle_deg = np.degrees(seg['theta'])
        x_approx = seg['r'] * np.cos(seg['theta'])
        y_approx = seg['r'] * np.sin(seg['theta'])

        print(f"   Ubicación:     Radio = {seg['r']:.1f}m, Ángulo = {angle_deg:.0f}° (X≈{x_approx:.1f}m, Y≈{y_approx:.1f}m)")
        print(f"   Puntos:        {seg['n_points']} puntos clasificados como SUELO")
        print(f"   Rango Z:       {seg['z_min']:.2f}m a {seg['z_max']:.2f}m")
        print(f"   Variación ΔZ:  {seg['delta_z']:.2f}m  ⚠️  {'ALTA VERTICALIDAD' if seg['delta_z'] > 1.0 else 'Moderada'}")
        print()

        # Mostrar muestra de puntos
        print(f"   Muestra de puntos (primeros 5):")
        print(f"   {'X':>8} {'Y':>8} {'Z':>8}")
        print(f"   {'-'*8} {'-'*8} {'-'*8}")

        sample_pts = seg['points'][:5]
        for pt in sample_pts:
            print(f"   {pt[0]:>8.2f} {pt[1]:>8.2f} {pt[2]:>8.2f}")
        print()

    print("="*80)
    print("RESUMEN DEL PROBLEMA")
    print("="*80)
    print()
    print(f"❌ Patchwork++ clasificó {len(vertical_segments)} segmentos verticales como 'suelo'")
    print()

    total_wall_points = sum(seg['n_points'] for seg in vertical_segments)
    print(f"   Total de puntos en paredes: {total_wall_points:,} ({100*total_wall_points/len(ground):.1f}% del suelo)")
    print()
    print("   Estos segmentos tienen variación vertical significativa (ΔZ > 0.5m),")
    print("   lo que indica que NO son superficies horizontales (suelo),")
    print("   sino superficies verticales o inclinadas (paredes, edificios, vehículos).")
    print()
    print("="*80)
    print("SOLUCIÓN REQUERIDA")
    print("="*80)
    print()
    print("Para corregir este problema, debes implementar POST-PROCESAMIENTO:")
    print()
    print("1. Analizar los PLANOS generados por Patchwork++:")
    print("   - patchwork.getCenters() → centroides de planos por bin CZM")
    print("   - patchwork.getNormals() → vectores normales de cada plano")
    print()
    print("2. Rechazar planos con normales HORIZONTALES (nz < 0.7):")
    print("   - nz = componente vertical de la normal")
    print("   - nz ≈ 1.0 → plano horizontal (suelo)")
    print("   - nz ≈ 0.0 → plano vertical (pared)")
    print()
    print("3. Validar con análisis geométrico local (KDTree):")
    print("   - Medir variación de altura (ΔZ) alrededor del centroide")
    print("   - Si ΔZ > 0.3m → CONFIRMAR como pared → RECHAZAR")
    print()
    print("4. Marcar bins rechazados como OBSTÁCULOS forzados:")
    print("   - Todos los puntos en esos bins → clasificar como obstáculo")
    print("   - Ignorar delta_r (distancia vs plano esperado)")
    print()
    print("Esta lógica ya está implementada en range_projection.py (líneas 394-427).")
    print("Necesitas portarla a paso_1.py para que funcione correctamente.")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
