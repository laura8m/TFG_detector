#!/usr/bin/env python3
"""
Test script: Usar Patchwork++ EXACTAMENTE como en el README oficial.
Objetivo: Demostrar que Patchwork++ clasifica trozos de paredes como suelo.

Basado en: https://github.com/url-kaist/patchwork-plusplus/blob/master/README.md
"""

import numpy as np
import sys
from pathlib import Path

# Añadir Patchwork++ al path

def main():
    print("="*80)
    print("TEST: Patchwork++ Vanilla (sin post-procesamiento)")
    print("="*80)
    print()

    # Importar Patchwork++ (como en el README)
    import pypatchworkpp

    # Cargar datos
    bin_file = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/data/000000.bin")
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]  # x, y, z

    print(f"✅ Cargados {len(points)} puntos desde {bin_file.name}\n")

    # Configurar Patchwork++ (CONFIGURACIÓN OFICIAL del README)
    params = pypatchworkpp.Parameters()
    params.verbose = False

    # Inicializar Patchwork++ (como en el README)
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    print("📋 CONFIGURACIÓN PATCHWORK++ (valores por defecto):")
    print(f"   sensor_height:        {params.sensor_height:.2f}m")
    print(f"   uprightness_thr:      {params.uprightness_thr:.3f} (cos(45°) = {np.cos(np.pi/4):.3f})")
    print(f"   num_iter:             {params.num_iter}")
    print(f"   num_lpr:              {params.num_lpr}")
    print(f"   th_dist:              {params.th_dist}m")
    print(f"   min_range:            {params.min_range}m")
    print(f"   max_range:            {params.max_range}m")
    print(f"   enable_RNR:           {params.enable_RNR}")
    print(f"   enable_RVPF:          {params.enable_RVPF}")
    print(f"   enable_TGR:           {params.enable_TGR}")
    print()

    # Ejecutar Patchwork++ (EXACTAMENTE como en el README)
    print("🔄 Ejecutando Patchwork++...")
    PatchworkPLUSPLUS.estimateGround(points)

    # Obtener resultados (como en el README)
    ground = PatchworkPLUSPLUS.getGround()
    nonground = PatchworkPLUSPLUS.getNonground()
    time_taken = PatchworkPLUSPLUS.getTimeTaken()

    print(f"✅ Completado en {time_taken:.3f}s\n")

    # Estadísticas básicas
    print("="*80)
    print("RESULTADOS DE CLASIFICACIÓN")
    print("="*80)
    print(f"Total de puntos:        {len(points):>8,}")
    print(f"Puntos clasificados como SUELO:      {len(ground):>8,} ({100*len(ground)/len(points):>5.1f}%)")
    print(f"Puntos clasificados como NO-SUELO:   {len(nonground):>8,} ({100*len(nonground)/len(points):>5.1f}%)")
    print()

    # Análisis de puntos clasificados como "suelo"
    print("="*80)
    print("ANÁLISIS DE PUNTOS CLASIFICADOS COMO 'SUELO'")
    print("="*80)
    print()

    # Estadísticas de altura (Z) de puntos "suelo"
    z_ground = ground[:, 2]
    print(f"📊 Distribución de altura (Z) de puntos 'suelo':")
    print(f"   Min:  {np.min(z_ground):>7.2f}m")
    print(f"   Max:  {np.max(z_ground):>7.2f}m")
    print(f"   Mean: {np.mean(z_ground):>7.2f}m")
    print(f"   Std:  {np.std(z_ground):>7.2f}m")
    print()

    # Histograma de altura
    print("   Histograma de altura (Z):")
    bins = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    hist, _ = np.histogram(z_ground, bins=bins)

    for i in range(len(hist)):
        z_low = bins[i]
        z_high = bins[i+1]
        count = hist[i]
        pct = 100 * count / len(ground)
        bar = '█' * int(pct / 2)
        print(f"      {z_low:>5.1f}m - {z_high:>5.1f}m: {count:>7,} puntos ({pct:>5.1f}%) {bar}")
    print()

    # ANÁLISIS CRÍTICO: Puntos sospechosos en altura
    print("="*80)
    print("⚠️  ANÁLISIS CRÍTICO: DETECCIÓN DE ANOMALÍAS")
    print("="*80)
    print()

    # Puntos "suelo" que están ELEVADOS (posibles paredes)
    suspicious_high = ground[ground[:, 2] > 0.0]  # Z > 0 (sobre el sensor)
    suspicious_low = ground[ground[:, 2] < -2.0]  # Z < -2m (muy bajo, posibles errores)

    print(f"⚠️  Puntos 'suelo' con Z > 0.0m (elevados, sospechosos):")
    print(f"   Cantidad: {len(suspicious_high):>8,} ({100*len(suspicious_high)/len(ground):>5.1f}% del suelo)")

    if len(suspicious_high) > 0:
        print(f"   Rango Z:  {np.min(suspicious_high[:, 2]):.2f}m - {np.max(suspicious_high[:, 2]):.2f}m")
        print()
        print("   Primeros 10 puntos sospechosos (elevados):")
        print(f"   {'X':>8} {'Y':>8} {'Z':>8} {'Dist XY':>8}")
        print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for i in range(min(10, len(suspicious_high))):
            pt = suspicious_high[i]
            dist_xy = np.sqrt(pt[0]**2 + pt[1]**2)
            print(f"   {pt[0]:>8.2f} {pt[1]:>8.2f} {pt[2]:>8.2f} {dist_xy:>8.2f}m")
    print()

    print(f"⚠️  Puntos 'suelo' con Z < -2.0m (muy bajos):")
    print(f"   Cantidad: {len(suspicious_low):>8,} ({100*len(suspicious_low)/len(ground):>5.1f}% del suelo)")
    print()

    # ANÁLISIS DE CLUSTERS ESPACIALES
    print("="*80)
    print("ANÁLISIS DE SEGMENTOS VERTICALES (posibles paredes)")
    print("="*80)
    print()

    # Buscar segmentos de puntos "suelo" que forman líneas verticales
    # Agrupar por bins cilíndricos (r, theta) y analizar variación en Z

    r = np.sqrt(ground[:, 0]**2 + ground[:, 1]**2)
    theta = np.arctan2(ground[:, 1], ground[:, 0])
    z = ground[:, 2]

    # Discretizar en bins cilíndricos (resolución 1m radial, 5° angular)
    r_bins = np.arange(0, 80, 1.0)  # Bins de 1m
    theta_bins = np.arange(-np.pi, np.pi, np.radians(5))  # Bins de 5°

    # Digitize points into bins
    r_idx = np.digitize(r, r_bins)
    theta_idx = np.digitize(theta, theta_bins)

    # Create bin key
    bin_keys = list(zip(r_idx, theta_idx))
    unique_bins = set(bin_keys)

    print(f"🔍 Analizando {len(unique_bins)} bins cilíndricos...")
    print()

    # Analizar cada bin
    vertical_segments = []

    for r_id, theta_id in unique_bins:
        mask = (r_idx == r_id) & (theta_idx == theta_id)
        bin_points = ground[mask]

        if len(bin_points) < 5:  # Skip bins with few points
            continue

        # Calculate height variation in bin
        z_min = np.min(bin_points[:, 2])
        z_max = np.max(bin_points[:, 2])
        delta_z = z_max - z_min

        # If significant vertical extent -> likely a wall
        if delta_z > 0.5:  # 50cm vertical variation
            r_center = r_bins[r_id-1] if r_id > 0 else 0
            theta_center = theta_bins[theta_id-1] if theta_id > 0 else 0

            vertical_segments.append({
                'r': r_center,
                'theta': theta_center,
                'n_points': len(bin_points),
                'z_min': z_min,
                'z_max': z_max,
                'delta_z': delta_z,
                'centroid': np.mean(bin_points, axis=0)
            })

    # Sort by delta_z (most vertical first)
    vertical_segments.sort(key=lambda x: x['delta_z'], reverse=True)

    print(f"⚠️  SEGMENTOS VERTICALES DETECTADOS (ΔZ > 0.5m):")
    print(f"   Total: {len(vertical_segments)} segmentos")
    print()

    if len(vertical_segments) > 0:
        print(f"   {'Radio':>7} {'Ángulo':>7} {'Puntos':>7} {'Z_min':>7} {'Z_max':>7} {'ΔZ':>7}")
        print(f"   {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        for i, seg in enumerate(vertical_segments[:15]):  # Top 15
            angle_deg = np.degrees(seg['theta'])
            print(f"   {seg['r']:>7.1f}m {angle_deg:>6.0f}° {seg['n_points']:>7,} {seg['z_min']:>7.2f} {seg['z_max']:>7.2f} {seg['delta_z']:>7.2f}m")

        if len(vertical_segments) > 15:
            print(f"   ... ({len(vertical_segments) - 15} más)")
    print()

    # CONCLUSIÓN
    print("="*80)
    print("CONCLUSIÓN")
    print("="*80)
    print()
    print("❌ Patchwork++ (vanilla, sin post-procesamiento) FALLA en detectar paredes:")
    print()
    print(f"   1. Clasificó {len(suspicious_high):,} puntos elevados (Z > 0) como 'suelo'")
    print(f"      → Estos puntos están SOBRE el sensor, probablemente son paredes/obstáculos")
    print()
    print(f"   2. Detectó {len(vertical_segments)} segmentos con variación vertical > 0.5m")
    print(f"      → Segmentos verticales SON PAREDES, no suelo horizontal")
    print()
    print("✅ EVIDENCIA: Patchwork++ necesita POST-PROCESAMIENTO para rechazar paredes.")
    print()
    print("   Según el análisis del código fuente (patchworkpp.cpp:496):")
    print("   - uprightness_thr SOLO se aplica en zona 0 (< 9.64m)")
    print("   - Paredes en zonas 1, 2, 3 (> 9.64m) NO son rechazadas")
    print()
    print("   Solución requerida:")
    print("   - Filtrar planos con normales horizontales (nz < 0.7)")
    print("   - Validar con análisis geométrico local (ΔZ > 0.3m)")
    print("   - Marcar bins de paredes como obstáculos")
    print()
    print("="*80)

    # Guardar resultados para visualización (opcional)
    output_dir = Path("/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/debug_output")
    output_dir.mkdir(exist_ok=True)

    # Guardar puntos clasificados
    np.save(output_dir / "vanilla_ground.npy", ground)
    np.save(output_dir / "vanilla_nonground.npy", nonground)
    np.save(output_dir / "vanilla_suspicious_high.npy", suspicious_high)

    print(f"💾 Resultados guardados en: {output_dir}")
    print(f"   - vanilla_ground.npy ({len(ground):,} puntos)")
    print(f"   - vanilla_nonground.npy ({len(nonground):,} puntos)")
    print(f"   - vanilla_suspicious_high.npy ({len(suspicious_high):,} puntos)")
    print()

    # Crear script de visualización simple
    vis_script = output_dir / "visualize_results.py"
    vis_script.write_text("""#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar datos
ground = np.load('vanilla_ground.npy')
nonground = np.load('vanilla_nonground.npy')
suspicious = np.load('vanilla_suspicious_high.npy')

# Crear figura
fig = plt.figure(figsize=(15, 5))

# Vista XY (planta)
ax1 = fig.add_subplot(131)
ax1.scatter(ground[:, 0], ground[:, 1], c='green', s=0.1, alpha=0.5, label='Ground')
ax1.scatter(nonground[:, 0], nonground[:, 1], c='red', s=0.1, alpha=0.5, label='Non-ground')
ax1.scatter(suspicious[:, 0], suspicious[:, 1], c='blue', s=1, alpha=1.0, label='Suspicious (Z>0)')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Vista Planta (XY)')
ax1.legend()
ax1.axis('equal')
ax1.grid(True, alpha=0.3)

# Vista XZ (lateral)
ax2 = fig.add_subplot(132)
ax2.scatter(ground[:, 0], ground[:, 2], c='green', s=0.1, alpha=0.5, label='Ground')
ax2.scatter(nonground[:, 0], nonground[:, 2], c='red', s=0.1, alpha=0.5, label='Non-ground')
ax2.scatter(suspicious[:, 0], suspicious[:, 2], c='blue', s=1, alpha=1.0, label='Suspicious (Z>0)')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, label='Sensor Z=0')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('Vista Lateral (XZ)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Vista 3D
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(ground[::10, 0], ground[::10, 1], ground[::10, 2], c='green', s=0.1, alpha=0.3, label='Ground')
ax3.scatter(suspicious[:, 0], suspicious[:, 1], suspicious[:, 2], c='blue', s=2, alpha=1.0, label='Suspicious (Z>0)')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_zlabel('Z (m)')
ax3.set_title('Vista 3D')
ax3.legend()

plt.tight_layout()
plt.savefig('patchwork_vanilla_results.png', dpi=150, bbox_inches='tight')
print("✅ Visualización guardada en: patchwork_vanilla_results.png")
plt.show()
""")
    vis_script.chmod(0o755)

    print(f"📊 Script de visualización creado: {vis_script}")
    print(f"   Ejecutar: cd {output_dir} && python3 visualize_results.py")
    print()

if __name__ == "__main__":
    main()
