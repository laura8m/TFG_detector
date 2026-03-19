# RESUMEN: Evaluación de range_projection.py

**Fecha**: 10 Marzo 2026
**Objetivo**: Analizar el desempeño de `range_projection.py` (implementación de referencia) vs `lidar_pipeline_suite.py`

---

## 1. Métricas de detección (Scan 0, SemanticKITTI 04)

### Comparación de rendimiento:

| Implementación | Recall | Precision | F1 | FPs | Detecciones |
|----------------|--------|-----------|----|----|-------------|
| **lidar_pipeline_suite Stage 2** | 91.60% | 63.48% | 74.99% | 12,857 | 35,202 |
| **lidar_pipeline_suite Stage 3** (CLOSEST WINS) | 43.09% | 67.81% | 52.70% | 4,990 | 15,502 |
| **range_projection.py** (P>0.5) | **86.09%** | 37.48% | 52.23% | **35,032** | **56,034** |

---

## 2. Hallazgos clave

### ✅ range_projection.py NO sufre tanto la pérdida de recall

- **Recall 86.09%** vs nuestro Stage 3: **43.09%**
- Diferencia: +43 puntos porcentuales
- **Conclusión**: `range_projection.py` mitiga parcialmente el problema de compresión 20:1

### ✗ Pero tiene MUCHO peor precision

- **Precision 37.48%** vs nuestro Stage 3: **67.81%**
- Diferencia: -30.3 puntos porcentuales
- **35,032 False Positives** (2.7× más que nuestro Stage 2)

### Trade-off diferente

- **range_projection.py**: Recall alto, Precision baja → threshold permisivo
- **lidar_pipeline_suite Stage 3**: Recall bajo, Precision alta → threshold conservador

---

## 3. Análisis técnico: ¿Por qué range_projection.py tiene mejor recall?

Revisando el código y los resultados:

### Diferencia 1: **Spatial Smoothing** (Inter-ring consistency)

**range_projection.py** (líneas 782-790):
```python
# 3. Spatial Smoothing (Inter-ring Consistency) on BOOSTED belief
P_final = self.apply_spatial_smoothing(P_belief)

# apply_spatial_smoothing usa cv2.bilateralFilter o cv2.GaussianBlur
# Esto PROPAGA probabilidad de obstacles a píxeles vecinos
```

**lidar_pipeline_suite.py**: NO implementa spatial smoothing explícito en Stage 3

**Efecto**: El smoothing hace que obstacles "sangren" a píxeles vecinos, aumentando recall pero reduciendo precision.

---

### Diferencia 2: **Shadow Boost** (Geometric validation)

**range_projection.py** (líneas 740-748):
```python
# SHADOW BOOST (Geometric Validation)
shadow_boost = self.detect_geometric_shadows(...)
self.belief_map += shadow_boost  # Boost ANTES de smoothing

# detect_geometric_shadows proyecta sombras detrás de obstacles
# y AUMENTA belief (boost +2.0) si hay vacío detrás
```

**lidar_pipeline_suite.py**: Implementa shadow validation pero solo en Stage 4 (después de Bayes Filter)

**Efecto**: Shadow boost **antes** del smoothing hace que la señal se propague más, aumentando recall.

---

### Diferencia 3: **Threshold de probabilidad**

**range_projection.py**:
```python
# Threshold: P > 0.5 → obstacle
# Log-odds equivalente: belief > log(0.5/0.5) = 0.0
```

**lidar_pipeline_suite.py**:
```python
# Threshold: P > 0.35 → obstacle (más permisivo en teoría)
# Log-odds equivalente: belief > log(0.35/0.65) ≈ -0.619
```

Pero en práctica, `range_projection.py` tiene **threshold efectivo más bajo** debido a:
- Spatial smoothing que eleva P en píxeles vecinos
- Shadow boost que incrementa belief antes del threshold

---

## 4. Causa raíz del problema de compresión 20:1

El problema persiste en ambas implementaciones, pero con diferentes síntomas:

### range_projection.py:
- **Compresión 20:1 SÍ ocurre**: 124k puntos → 6k píxeles ocupados
- **Mitigation**: Spatial smoothing + Shadow boost compensan parcialmente
- **Resultado**: Recall 86% (aceptable) pero Precision 37% (inaceptable)

### lidar_pipeline_suite.py Stage 3:
- **Compresión 20:1 SÍ ocurre**: igual que range_projection
- **Sin mitigation**: No spatial smoothing, shadow validation después
- **Resultado**: Recall 43% (inaceptable) pero Precision 68% (aceptable)

**Conclusión**: El problema de "closest wins" es **inevitable** con range image. Solo puedes elegir el trade-off.

---

## 5. Recomendaciones

### Corto plazo (1-2 días):

1. **Implementar Spatial Smoothing en lidar_pipeline_suite.py**:
   ```python
   # Después de Bayes Filter, antes de threshold
   belief_prob_smooth = cv2.bilateralFilter(
       belief_prob.astype(np.float32),
       d=5,  # Neighborhood size
       sigmaColor=0.1,
       sigmaSpace=1.5
   )
   ```
   **Efecto esperado**: Recall 43% → ~70-80%, Precision 68% → ~50-55%

2. **Mover Shadow Validation ANTES del Bayes Filter** (como range_projection.py):
   - Calcular shadow boost en Stage 2
   - Agregar a likelihood antes de proyectar a range image

### Medio plazo (1 semana):

3. **Implementar Stage 3 per-point** (sin range image):
   - Evitar compresión 20:1 completamente
   - Usar KDTree para asociación temporal
   - **Efecto esperado**: Recall >90%, Precision >60%

### Largo plazo (1 mes):

4. **Entrenar CNN** (como Dewan et al.):
   - Dataset: SemanticKITTI (43k scans etiquetados)
   - Arquitectura: ResNet18 adaptado a LiDAR
   - **Solución definitiva**: Recall >95%, Precision >70%

---

## 6. Código ejecutado

### Scripts creados:

1. **launch_range_projection.sh**: Wrapper para ejecutar `range_projection.py` con ROS 2 Jazzy
2. **analyze_range_projection_output.py**: Analiza `belief_prob.npy` y calcula métricas
3. **evaluate_range_projection.py**: Script de análisis conceptual

### Modificaciones a range_projection.py:

Línea 328: Arreglado bug de `len(None)`
```python
# Antes:
if hasattr(self, 'poses') and len(self.poses) > self.current_scan:

# Después:
if hasattr(self, 'poses') and self.poses is not None and len(self.poses) > self.current_scan:
```

Línea 785: Guardar belief_prob para evaluación
```python
self.belief_prob = P_final  # (H, W) probabilidad final después de smoothing
```

Líneas 2244-2293: Método `save_evaluation_metrics()` para exportar resultados

### Ejecución:

```bash
# Lanzar range_projection.py
bash launch_range_projection.sh 0 0

# Analizar resultados
python3.12 tests/analyze_range_projection_output.py --scan 0
```

### Resultados guardados:

- `/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/tests/range_projection_output/belief_prob_scan_0.npy`

---

## 7. Conclusión final

### Pregunta original:
> "¿Por qué range_projection.py funciona mejor que lidar_pipeline_suite.py Stage 3?"

### Respuesta:
**NO funciona "mejor"**, tiene un **trade-off diferente**:

| Aspecto | range_projection.py | lidar_pipeline_suite Stage 3 |
|---------|---------------------|-------------------------------|
| **Recall** | 86.09% ✅ | 43.09% ✗ |
| **Precision** | 37.48% ✗ | 67.81% ✅ |
| **F1 Score** | 52.23% | 52.70% (similar) |
| **False Positives** | 35,032 ✗✗ | 4,990 ✅ |

**Ventaja de range_projection.py**: Spatial smoothing + Shadow boost **antes** del threshold
**Desventaja de range_projection.py**: **Precision inaceptable** para aplicaciones reales (37%)

### Solución óptima:

**Implementar Stage 3 per-point** (sin range image) para evitar compresión 20:1 completamente.

**Recall esperado**: >90%
**Precision esperada**: >60%
**F1 esperado**: >72%

---

**Documentado por**: Claude Code
**Fecha**: 10 Marzo 2026
**Archivos relacionados**:
- `range_projection.py` (modificado)
- `tests/analyze_range_projection_output.py` (nuevo)
- `tests/evaluate_range_projection.py` (nuevo)
- `launch_range_projection.sh` (nuevo)
- `tests/RESUMEN_PROBABILIDAD_BINARIA_Y_COMPRESION_20_1.md` (sesión anterior)
