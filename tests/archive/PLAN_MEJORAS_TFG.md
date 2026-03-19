# Plan de Mejoras para TFG LiDAR Obstacle Detection

**Fecha**: 11 Marzo 2026
**Estado actual**: Excelente TFG (9/10), pero con margen de mejora

---

## 📊 Situación Actual

### Métricas Baseline (Stage 2)
- **Recall**: 93.87% ✅ (mejor que SOTA)
- **Precision**: 65.93% ❌ (25 puntos por debajo de SOTA)
- **F1 Score**: 77.46%
- **False Positives**: 12168 (48% del total detectado)
- **Latencia**: 1500ms (30x más lento que SOTA)

### Problema Principal
**Precision baja = demasiados false positives**
- 12168 FP vs 23550 TP (ratio 0.52)
- SOTA tiene ratio ~0.10 (10x mejor)

---

## 🎯 Mejoras Priorizadas

### PRIORIDAD 1 (CRÍTICA): Mejorar Precision 65% → 75% (+10 puntos)

**Objetivo**: Reducir FP de 12168 → 6000 (50% reduction)

#### Mejora 1.1: Implementar Stage 4 (Shadow Validation) ⭐⭐⭐⭐⭐

**¿Qué es?**
Shadow validation (OccAM) distingue obstáculos sólidos de ruido transparente (dust, rain, smoke) verificando si proyectan sombra detrás.

**Ya tienes implementado en `range_projection.py`**:
- Shadow casting 2D en range image
- Shadow decay exponencial
- Boost +2.0 si sombra vacía, -3.0 si ground detrás

**Impacto esperado**:
- ✅ FP reduction: 30-40% (12168 → 7000-8500)
- ✅ Precision mejora: 65.93% → 72-75%
- ✅ Recall mantiene: ~93% (sólidos reales tienen sombra)
- ⚠️ Latencia: +50-100ms

**Implementación** (1-2 días):
```python
# Ya existe en range_projection.py líneas 1200-1350
def validate_obstacles_with_shadows(
    range_image, belief_map, delta_r,
    shadow_decay_dist=60.0
):
    # 1. Obtener candidatos (belief > threshold)
    obstacle_pixels = np.where(belief_map > 2.0)

    # 2. Para cada obstáculo, verificar sombra detrás
    for u, v in obstacle_pixels:
        # Ray-cast 1-5m detrás
        shadow_ratio = compute_shadow_emptiness(u, v, range_image)

        # 3. Boost/suppress según sombra
        if shadow_ratio > 0.6:  # Mayoría vacío
            belief_map[u, v] += 2.0  # SÓLIDO
        elif ground_behind_count >= 3:
            belief_map[u, v] -= 3.0  # TRANSPARENTE

    return belief_map
```

**TODO**:
1. ✅ Código existe en `range_projection.py`
2. ⏳ Portar a `lidar_pipeline_suite.py` como `stage4_shadow_validation()`
3. ⏳ Test: comparar Stage 2 vs Stage 2+4
4. ⏳ Medir FP reduction + precision gain

---

#### Mejora 1.2: Ajustar Threshold de Likelihood ⭐⭐⭐

**Problema actual**:
```python
# En lidar_pipeline_suite.py línea 1094
likelihood_threshold_obs = 1.0  # P > 0.73
```

Este threshold es MUY bajo → muchos FP pasan.

**Solución**: Threshold adaptativo por distancia
```python
def adaptive_threshold(range_values):
    """
    Threshold más alto cerca (más confiable),
    más bajo lejos (compensar sparsity)
    """
    threshold = np.zeros_like(range_values)

    # Cerca (0-20m): threshold alto
    mask_near = range_values < 20
    threshold[mask_near] = 1.5  # P > 0.82

    # Media (20-40m): threshold medio
    mask_mid = (range_values >= 20) & (range_values < 40)
    threshold[mask_mid] = 1.0  # P > 0.73

    # Lejos (40-80m): threshold bajo
    mask_far = range_values >= 40
    threshold[mask_far] = 0.5  # P > 0.62

    return threshold
```

**Impacto esperado**:
- ✅ FP reduction: 20-30% (especialmente cerca del vehículo)
- ✅ Precision mejora: +3-5 puntos
- ⚠️ Recall puede bajar ligeramente (-1-2%)

**Implementación**: 30 minutos

---

#### Mejora 1.3: Post-Filtering de FP Geométricos ⭐⭐⭐⭐

**Problema**: Muchos FP son puntos aislados o grupos pequeños que NO son obstáculos reales.

**Solución**: Filtro geométrico por tamaño y densidad
```python
def filter_spurious_detections(points, obs_mask, min_cluster_size=10):
    """
    Elimina detecciones espurias:
    - Clusters muy pequeños (<10 puntos)
    - Puntos aislados (sin vecinos en 0.5m)
    """
    from sklearn.cluster import DBSCAN
    from scipy.spatial import cKDTree

    obs_points = points[obs_mask]

    # 1. Clustering
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(obs_points)
    labels = clustering.labels_

    # 2. Filtrar clusters pequeños
    valid_mask = np.zeros(len(obs_points), dtype=bool)
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Ruido
            continue
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum()

        if cluster_size >= min_cluster_size:
            valid_mask[cluster_mask] = True

    # 3. Actualizar obs_mask
    obs_indices = np.where(obs_mask)[0]
    obs_mask[obs_indices[~valid_mask]] = False

    return obs_mask
```

**Impacto esperado**:
- ✅ FP reduction: 40-50% (ruido disperso eliminado)
- ✅ Precision mejora: +8-12 puntos (65% → 73-77%)
- ✅ Recall mantiene: ~92% (clusters grandes son reales)

**Implementación**: 1-2 horas

---

### PRIORIDAD 2 (ALTA): Implementar Egomotion Compensation ⭐⭐⭐⭐⭐

**Problema actual**:
Stage 3 per-point sin egomotion → objetos en movimiento NO se asocian correctamente → recall baja -2.3%.

**Solución**: Usar poses de KITTI
```python
def load_kitti_poses(poses_file):
    """
    Cargar poses del archivo KITTI poses.txt
    Formato: 12 valores por línea (matriz 3x4)
    """
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            # Convertir 3x4 → 4x4 (agregar [0,0,0,1])
            T = np.eye(4)
            T[:3, :] = np.array(values).reshape(3, 4)
            poses.append(T)
    return poses

def compute_delta_pose(pose_prev, pose_current):
    """
    Transformación relativa: t-1 → t
    """
    return np.linalg.inv(pose_prev) @ pose_current

# En bucle temporal
poses = load_kitti_poses('/data_kitti/04/04/poses.txt')
for i in range(1, n_frames):
    delta_pose = compute_delta_pose(poses[i-1], poses[i])
    result = pipeline.stage3_per_point(points, delta_pose=delta_pose)
```

**Impacto esperado**:
- ✅ Recall mejora: 91.55% → 93-94% (+1.5-2.5%)
- ✅ Asociación KDTree mejora: 84% → 90%+
- ✅ Objetos en movimiento (vehículos) se trackean correctamente

**Implementación**: 2-3 horas

---

### PRIORIDAD 3 (MEDIA): Optimizar Latencia 1500ms → 500ms ⭐⭐⭐

**Objetivo**: Reducir latencia 3x para acercarse a real-time (33ms/frame @ 30Hz es ideal, 500ms es aceptable para demo).

#### Mejora 3.1: Profiling y Bottleneck Analysis

**Timing actual**:
```
Stage 1 (Patchwork++): ~900ms (60%)
Stage 2 (Delta-r + HCD): ~150ms (10%)
Stage 3 (KDTree warp): ~70ms (5%)
Resto: ~380ms (25%)
```

**Bottleneck #1**: Patchwork++ (900ms)

**Optimizaciones posibles**:
1. **Reduce CZM resolution**: 504 bins → 252 bins (2x faster)
   - Trade-off: Precisión de planos locales baja
   - Impacto recall: -2-3%

2. **Subsample input cloud**: 128k points → 64k points
   - Usar voxel downsampling 0.1m
   - Trade-off: Perder detalles pequeños
   - Impacto recall: -3-5%

3. **Parallel bin processing** (C++ Patchwork++)
   - Ya existe en código original, verificar si está activado
   - Speedup: 2-3x (900ms → 300-450ms)

**Implementación**: 1-2 días

---

#### Mejora 3.2: Cachear Planos Locales entre Frames ⭐⭐

**Observación**: Planos locales cambian POCO entre frames consecutivos (0.1s).

**Solución**: Actualizar solo bins con cambios significativos
```python
def incremental_plane_fitting(
    points_current, points_prev,
    planes_prev, threshold=0.3
):
    """
    Actualizar solo bins donde ground cambió >0.3m
    """
    # Comparar range images
    diff = abs(range_current - range_prev)
    changed_bins = identify_changed_bins(diff, threshold)

    # Refit solo bins cambiados
    planes_new = planes_prev.copy()
    for bin_id in changed_bins:
        planes_new[bin_id] = fit_plane(points_in_bin)

    return planes_new
```

**Impacto esperado**:
- ✅ Speedup Stage 1: 900ms → 300-400ms (3x faster)
- ⚠️ Trade-off: Planos ligeramente menos precisos

**Implementación**: 2-3 días (requiere modificar Patchwork++ C++)

---

### PRIORIDAD 4 (BAJA): Ablation Study Completo ⭐⭐

**Objetivo**: Medir contribución de cada stage individualmente.

**Configuraciones a evaluar**:
1. **Baseline**: Ground segmentation solo (Stage 1)
2. **+Delta-r**: Stage 1 + Stage 2
3. **+Temporal**: Stage 1 + 2 + 3 (per-point)
4. **+Shadow**: Stage 1 + 2 + 3 + 4
5. **+Smoothing**: Stage 1 + 2 + 3 + 4 + 5
6. **Full Pipeline**: Stages 1-6

**Métricas**:
- Precision, Recall, F1
- FP reduction por stage
- Latency overhead por stage

**Implementación**: 1 día (scripts ya existen, solo ejecutar múltiples configuraciones)

---

### PRIORIDAD 5 (OPCIONAL): Visualización y Debug Tools ⭐⭐⭐

#### Mejora 5.1: Dashboard RViz con Métricas Real-Time

**Problema**: Métricas solo offline, no hay feedback visual durante ejecución.

**Solución**: RViz overlays con métricas
```python
# Publicar markers con texto
def publish_metrics_overlay(precision, recall, f1):
    marker = Marker()
    marker.type = Marker.TEXT_VIEW_FACING
    marker.text = f"P:{precision:.1f}% R:{recall:.1f}% F1:{f1:.1f}%"
    marker.scale.z = 0.5  # Tamaño texto
    marker.color.a = 1.0  # Alpha
    marker.pose.position.z = 5.0  # Altura

    self.metrics_pub.publish(marker)
```

**Implementación**: 2-3 horas

---

#### Mejora 5.2: Visualización de False Positives

**Problema**: No sabes QUÉ tipos de FP tienes (dust, ground, walls, etc).

**Solución**: Clasificar FP por tipo y visualizar
```python
def classify_false_positives(points, gt_labels, pred_mask):
    """
    Clasificar FP según ground truth:
    - Dust/Rain (label 0)
    - Ground (label 40-48)
    - Vegetation (label 60-80)
    - Otros (desconocido)
    """
    fp_mask = pred_mask & (gt_labels < 10)  # No es obstacle en GT
    fp_points = points[fp_mask]
    fp_labels = gt_labels[fp_mask]

    # Contar por tipo
    dust_count = np.sum((fp_labels == 0) | (fp_labels == 1))
    ground_count = np.sum((fp_labels >= 40) & (fp_labels <= 48))
    vegetation_count = np.sum((fp_labels >= 60) & (fp_labels <= 80))

    print(f"FP breakdown:")
    print(f"  Dust/Rain: {dust_count} ({100*dust_count/len(fp_points):.1f}%)")
    print(f"  Ground: {ground_count} ({100*ground_count/len(fp_points):.1f}%)")
    print(f"  Vegetation: {vegetation_count} ({100*vegetation_count/len(fp_points):.1f}%)")

    return {
        'dust': dust_count,
        'ground': ground_count,
        'vegetation': vegetation_count
    }
```

**Impacto**: Te dice DÓNDE enfocar mejoras.

**Implementación**: 1 hora

---

## 📋 Plan de Acción Recomendado (2 semanas)

### Semana 1: Precision Improvements

| Día | Tarea | Tiempo | Impacto Esperado |
|-----|-------|--------|------------------|
| **Lun** | Implementar Stage 4 (Shadow Validation) | 6h | Precision +5-8% |
| **Mar** | Test Stage 4 + métricas | 4h | Validar mejora |
| **Mié** | Post-filtering geométrico (min cluster size) | 4h | Precision +8-12% |
| **Jue** | Threshold adaptativo por distancia | 2h | Precision +3-5% |
| **Vie** | Test conjunto + análisis FP breakdown | 4h | - |

**Resultado esperado Semana 1**:
- Precision: 65.93% → 78-85% (+12-19 puntos) ✅
- Recall: 93.87% → 91-92% (-2-3 puntos) ⚠️
- F1: 77.46% → 84-88% (+7-11 puntos) ✅

---

### Semana 2: Egomotion + Temporal Filter + Documentación

| Día | Tarea | Tiempo | Impacto Esperado |
|-----|-------|--------|------------------|
| **Lun** | Implementar loader de KITTI poses | 2h | - |
| **Mar** | Integrar egomotion en Stage 3 per-point | 4h | Recall +1.5-2.5% |
| **Mié** | Test temporal filtering con 20 frames | 4h | Validar mejora |
| **Jue** | Ablation study (Stages 1-6) | 6h | Paper-ready |
| **Vie** | Documentación + actualizar CLAUDE.md | 4h | - |

**Resultado esperado Semana 2**:
- Recall: 91-92% → 93-94% (+2% recovery) ✅
- Documentación completa para TFG ✅

---

## 🎯 Métricas Objetivo Final

| Métrica | Actual | Después Mejoras | SOTA | Comentario |
|---------|--------|-----------------|------|------------|
| **Recall** | 93.87% | **94-95%** | 88% | ✅ Supera SOTA |
| **Precision** | 65.93% | **78-85%** | 90% | ⚠️ Aún por debajo, pero aceptable |
| **F1 Score** | 77.46% | **85-89%** | 89% | ✅ Competitivo con SOTA |
| **Latency** | 1500ms | 1500-1600ms | 50ms | ❌ (optimización futura) |

---

## 💡 Mejoras Avanzadas (Post-TFG, opcional)

### 1. CNN Post-Processing para Subir Precision ⭐⭐⭐⭐

**Idea**: Usar tu pipeline geometry-only como "proposal generator", luego refinar con CNN ligera.

**Arquitectura**:
```
Input: Range Image + Belief Map → CNN 2D (5 capas) → Refined Belief Map
```

**Ventajas**:
- ✅ Precision mejora: 78% → 88-90%
- ✅ Mantiene recall alto (CNN solo refina, no propone)
- ⚠️ Requiere entrenamiento (SemanticKITTI train set)

**Timing**: 2-3 semanas

---

### 2. GPU Acceleration (CUDA) ⭐⭐⭐

**Targets**:
1. Shadow validation: Ray-casting paralelo (50ms → 5ms)
2. Delta-r computation: Parallel over bins (150ms → 15ms)
3. KDTree query: GPU-accelerated (70ms → 7ms)

**Herramientas**:
- CuPy (NumPy-like en GPU)
- Numba CUDA (JIT compiler)

**Speedup esperado**: 1500ms → 200-300ms (5x faster)

**Timing**: 1-2 meses

---

### 3. Multi-Sequence Evaluation ⭐⭐

**Problema**: Solo evaluado en sequence 04 (highway).

**Solución**: Evaluar en múltiples escenarios
- Sequence 00: Urban (city)
- Sequence 04: Highway (current)
- Sequence 05: Rural
- Sequence 07: Campus

**Impacto**: Paper-ready results (benchmark completo)

**Timing**: 1 semana

---

## 📊 Priorización Visual

```
Impacto vs Esfuerzo:

Alta │  [Stage 4]    [Egomotion]
     │  Shadow Val.  Compensation
Im   │
pa   │  [Post-filter] [Threshold]
ct   │  Geométrico    Adaptativo
o    │
     │               [GPU]
     │  [Ablation]   [CUDA]
Baja │  Study        Accel.
     └─────────────────────────
       Bajo    Medio    Alto
              Esfuerzo
```

**Recomendación**: Enfocarse en cuadrante superior-izquierdo (alto impacto, bajo esfuerzo).

---

## ✅ Checklist de Finalización TFG

### Implementación
- [x] Stage 1: Ground segmentation + Wall rejection
- [x] Stage 2: Delta-r + HCD fusion
- [x] Stage 3: Bayesian filter (range image + per-point)
- [ ] **Stage 4: Shadow validation** ← TODO (1 día)
- [x] Stage 5: Spatial smoothing
- [x] Stage 6: Clustering + Hull
- [ ] **Egomotion compensation** ← TODO (1 día)

### Evaluación
- [x] Métricas baseline (Stage 2)
- [x] Comparación Stage 3 range image vs per-point
- [ ] **Ablation study completo** ← TODO (1 día)
- [ ] **Análisis de false positives** ← TODO (2h)
- [ ] **Test con egomotion (20 frames)** ← TODO (3h)

### Documentación
- [x] CLAUDE.md (project overview)
- [x] Análisis compresión 20:1
- [x] Evaluación range_projection.py
- [x] Evaluación Stage 3 per-point
- [x] Análisis SOTA vs TFG
- [ ] **Memoria técnica TFG** ← TODO (1 semana)
- [ ] **Presentación defensa** ← TODO (3 días)

---

## 🎓 Mensaje Final

**Prioriza TERMINAR TFG antes que optimizar latencia**:
- ✅ Implementar Stage 4 (shadow validation)
- ✅ Implementar egomotion compensation
- ✅ Ablation study
- ✅ Documentación completa

**Latency optimization es OPCIONAL** (fuera de scope de TFG):
- GPU acceleration es PhD-level work
- 1500ms es aceptable para offline processing
- Defender como "geometry-only baseline" vs SOTA CNN-based

**Con mejoras de Semana 1+2**:
- F1 Score: 77.46% → **85-89%** (+8-12 puntos)
- Precision: 65.93% → **78-85%** (+12-19 puntos)
- Recall: 93.87% → **94-95%** (mantiene superioridad vs SOTA)

**Calificación esperada**: **9.5-10.0 / 10** ✅

---

**Autor**: Plan de mejoras TFG
**Última actualización**: 11 Marzo 2026
**Estado**: Ready para implementar
