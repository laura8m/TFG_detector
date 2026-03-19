# Feedback Técnico - TFG Percepción LiDAR Off-Road

**Fecha**: 2026-03-02
**Contexto**: Sistema de detección de obstáculos positivos/negativos/voids con Patchwork++ y proyección de rango
**Dataset**: GOose (off-road)
**Performance objetivo**: <200ms por frame (i7-1255U)

---

## 🎯 Evaluación de tu enfoque actual

### ✅ Fortalezas de tu pipeline

1. **Auditoría de Patchwork++ correcta**
   - Has identificado el bug crítico: RVPF solo activo en zona 0 (línea 496 del código fuente)
   - Tu solución `nz < 0.7` está alineada con `uprightness_thr = 0.707` del código original
   - Implementación eficiente: filtrado es O(n_bins) ≈ O(274) → despreciable (<1ms)

2. **Metodología sólida**
   - Delta-r (rango esperado vs medido) es una métrica robusta para terrenos off-road
   - Separación ground/non-ground antes de análisis geométrico es correcto
   - Inspiración en ROBIO 2024 (T_height + T_var) es adecuada para tu caso

3. **Performance razonable**
   - 147ms para Patchwork++ en HDL-64E (~130k puntos) es aceptable
   - 197ms totales te deja margen para validación adicional (objetivo <200ms)

---

## ⚠️ Gaps críticos identificados

### 1. **Voids no están implementados**

**Problema**: Tu `paso_1.py` NO detecta discontinuidades de profundidad (voids).

**Por qué es crítico**: En entornos off-road (GOose dataset), los voids aparecen en:
- Bordes de rocas/troncos
- Transiciones terreno→precipicio
- Oclusiones dinámicas (vegetación)

**Solución implementada**: Ver `paso_1_extensions.py` → función `detect_voids()`

**Métricas clave**:
```python
void_threshold = 2.0m    # Salto de profundidad mínimo
var_threshold = 0.1m²    # Varianza máxima (diferencia de terreno irregular)
```

**Cómo integrar**:
```python
# En tu pipeline principal, añadir DESPUÉS de calcular delta_r:
variance = compute_local_variance(points, local_planes)
void_mask, void_clusters = detect_voids(points, delta_r, rejected_mask, variance)
```

---

### 2. **Obstáculos negativos sin validación**

**Problema**: Tu delta_r detecta `r_measured < r_expected`, pero no hay lógica para:
- Filtrar ruido (delta_r negativo pequeño)
- Validar clusters coherentes (baches reales vs sombras de rayos)

**Solución implementada**: Ver `paso_1_extensions.py` → función `detect_negative_obstacles()`

**Criterios de validación**:
1. `delta_r < -0.3m` (threshold configurable)
2. Cluster coherente: `min_cluster_size = 10 puntos`
3. Varianza moderada: `variance < 0.5m²` (diferencia de ruido)

**Ejemplo de uso**:
```python
negative_mask, negative_clusters = detect_negative_obstacles(
    points, delta_r, rejected_mask, variance,
    negative_threshold=-0.3,
    min_cluster_size=10
)
```

---

### 3. **Falta capa de integridad (ROBIO 2024)**

**Problema**: No tienes implementado el `T_var` (varianza local) que mencionas en tu descripción.

**Por qué es importante**: En horizontes lejanos (>50m), el delta_r es menos confiable debido a:
- Baja densidad de puntos
- Mayor error de medición del LiDAR
- Planos locales menos precisos

**Solución implementada**: Ver `paso_1_extensions.py` → función `compute_local_variance()`

**Cómo funciona**:
```python
# Calcula varianza de distancias al plano dentro de cada bin CZM
variance_per_point = compute_local_variance(points, local_planes)

# Alta varianza → zona sospechosa (requiere validación adicional)
# Baja varianza → alta confianza en clasificación
```

**Uso en tu pipeline**:
```python
# Penalizar detecciones en zonas de alta varianza
integrity_score = compute_integrity_score(points, delta_r, variance, rejected_mask)
final_obstacles = positive_mask & (integrity_score > 0.5)  # Filtro de confianza
```

---

## 📊 Comparación con tu implementación actual

| Componente | paso_1.py actual | Necesario para TFG | Implementado en extensions |
|------------|------------------|---------------------|---------------------------|
| Filtrado de paredes (`nz < 0.7`) | ✅ | ✅ | - |
| Delta-r básico | ✅ | ✅ | - |
| Varianza local (T_var) | ❌ | ✅ | ✅ `compute_local_variance()` |
| Detección de voids | ❌ | ✅ | ✅ `detect_voids()` |
| Obstáculos negativos con clustering | ❌ | ✅ | ✅ `detect_negative_obstacles()` |
| Score de integridad | ❌ | ⚠️ (recomendado) | ✅ `compute_integrity_score()` |

---

## 🚀 Plan de integración recomendado

### Opción 1: Integración modular (recomendada)

1. **Copiar funciones de `paso_1_extensions.py` a `paso_1.py`**
2. **Modificar tu función principal** para llamar a las nuevas funciones:

```python
def process_frame(self, scan_file):
    # ... código existente para cargar puntos ...

    # Segmentación de suelo (YA IMPLEMENTADO)
    ground_points, n_per_point, d_per_point, rejected_mask = self.segment_ground(points)

    # Delta-r (YA IMPLEMENTADO)
    r_expected = self.compute_expected_range(points, n_per_point, d_per_point)
    r_measured = np.linalg.norm(points, axis=1)
    delta_r = r_measured - r_expected

    # NUEVO: Varianza local
    variance = self.compute_local_variance(points, self.local_planes)

    # NUEVO: Obstáculos positivos (con umbral)
    positive_mask = (delta_r > 0.3) & ~rejected_mask

    # NUEVO: Obstáculos negativos
    negative_mask, negative_clusters = self.detect_negative_obstacles(
        points, delta_r, rejected_mask, variance
    )

    # NUEVO: Voids
    void_mask, void_clusters = self.detect_voids(
        points, delta_r, rejected_mask, variance
    )

    # NUEVO: Score de integridad
    integrity = self.compute_integrity_score(points, delta_r, variance, rejected_mask)

    # Visualización / guardado de resultados
    self.visualize_results(points, positive_mask, negative_mask, void_mask, integrity)
```

### Opción 2: Usar `process_frame_extended()` directamente

```python
# En tu script principal:
from paso_1_extensions import process_frame_extended

results = detector.process_frame_extended(points)

# Acceder a resultados:
positive_obs = results['positive_obstacles']
negative_obs = results['negative_obstacles']
voids = results['voids']
integrity = results['integrity_score']
```

---

## 📈 Estimación de impacto en performance

| Componente | Complejidad | Estimación tiempo (i7-1255U) |
|------------|-------------|------------------------------|
| `compute_local_variance()` | O(n_bins × n_points/bin) | ~5-10ms |
| `detect_voids()` (KDTree) | O(n × log n) | ~15-20ms |
| `detect_negative_obstacles()` (DBSCAN) | O(n × log n) | ~10-15ms |
| `compute_integrity_score()` | O(n) | ~1-2ms |
| **TOTAL OVERHEAD** | | **~30-50ms** |

**Proyección total**: 197ms (actual) + 50ms (nuevas funciones) = **~250ms**

⚠️ **EXCEDES tu objetivo de 200ms** → Opciones de optimización:

1. **Reducir resolución de imagen de rango** (si usas proyección esférica)
2. **Limitar detección de voids a horizonte cercano** (<30m)
3. **Paralelizar Patchwork++ y delta-r** (si es posible)
4. **Usar implementación C++ para funciones críticas** (KDTree, DBSCAN)

---

## 🔍 Validación experimental recomendada

Para validar tu pipeline en GOose dataset:

### Test 1: Obstáculos positivos
- **Ground truth**: Rocas, troncos, vegetación densa
- **Métrica esperada**: Recall > 85%, Precision > 80%
- **Fallos esperados**: Vegetación baja (delta_r < threshold)

### Test 2: Obstáculos negativos
- **Ground truth**: Baches, zanjas, hundimientos
- **Métrica esperada**: Recall > 70% (más difícil que positivos)
- **Fallos esperados**: Baches poco profundos (<0.2m)

### Test 3: Voids
- **Ground truth**: Bordes de obstáculos, oclusiones
- **Métrica esperada**: Detección de >90% de edges significativos (>2m salto)
- **Fallos esperados**: Falsos positivos en terreno muy irregular

### Test 4: Integridad en horizontes
- **Validación**: Comparar integrity_score vs error ground truth
- **Esperado**: Correlación negativa (score alto → error bajo)

---

## 💡 Recomendaciones finales

### ✅ Lo que debes hacer

1. **Integrar `compute_local_variance()`** → Es crítico para T_var de ROBIO 2024
2. **Implementar detección de voids** → Requisito explícito de tu TFG
3. **Validar con GOose ground truth** → Necesitas métricas cuantitativas
4. **Documentar performance** → Desglosar 197ms por componente

### ⚠️ Lo que puedes optimizar después (si excedes 200ms)

1. Usar implementación C++ para KDTree (scipy → Open3D/PCL)
2. Reducir eps de DBSCAN (menos iteraciones)
3. Submuestrear puntos lejanos (>50m) para voids

### ❌ Lo que NO debes hacer

1. **NO reimplementar Patchwork++** → 147ms es razonable, no es tu cuello de botella real
2. **NO eliminar filtrado de paredes** → Es tu contribución clave sobre vanilla Patchwork++
3. **NO usar solo delta_r sin varianza** → Tendrás muchos falsos positivos en terreno irregular

---

## 📚 Referencias para tu TFG

1. **Tu filtrado de paredes**: Citar línea 496 de `patchworkpp.cpp` + demostrar bug con EVIDENCIA_PATCHWORK.md
2. **Varianza local**: ROBIO 2024 (citar T_var methodology)
3. **Delta-r**: Similar a "Range residual" usado en LeGO-LOAM
4. **Detección de voids**: Similar a "depth discontinuity detection" en SegMap

---

## 🎓 Conclusión

**¿Estás cubriendo los inicios de tu idea en paso_1.py?**
**Respuesta**: **Sí, pero solo parcialmente** (60%).

- ✅ Tienes la base correcta (filtrado de paredes + delta_r)
- ⚠️ Te faltan componentes críticos (voids, negativos, T_var)
- ✅ Tu enfoque metodológico es sólido (auditoría de Patchwork++ + ROBIO 2024)

**Siguiente paso inmediato**: Integra las funciones de `paso_1_extensions.py` en tu pipeline y valida con **al menos 50 frames de GOose** para obtener métricas preliminares.

**Para tu defensa de TFG**: Enfatiza tu auditoría del código fuente de Patchwork++ (muy pocos trabajos hacen esto) y tu capa de validación de integridad (contribución sobre vanilla Patchwork++).

---

**¿Preguntas técnicas?**
- Implementación específica de alguna función
- Estrategias de optimización
- Validación experimental
