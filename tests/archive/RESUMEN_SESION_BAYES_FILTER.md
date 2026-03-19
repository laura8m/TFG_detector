# Resumen Sesión: Análisis Bayes Filter y Stage 3

**Fecha**: 2026-03-09
**Contexto**: Continuación de sesión previa sobre implementación de Stage 3 (Bayesian Temporal Filter)

## Problema Inicial

Stage 3 está matando el recall:
- **Stage 2 (per-point)**: 91.6% recall, 63.5% precision, F1=75.0% ✓✓✓
- **Stage 3 (Bayes Filter)**: 43.1% recall, 67.8% precision, F1=52.7% ✗✗✗

**Pérdida de recall**: -48.5 puntos porcentuales (de 91.6% → 43.1%)

---

## Trabajo Realizado

### 1. Lectura del Paper "DeepTemporalSeg" (Dewan et al., IROS 2018)

**Ecuación 9 del paper** (Bayes Filter con log-odds):

```
l_t(O_c^t) = log[P(O_c^t | ξ_c^t) / (1 - P(O_c^t | ξ_c^t))] + l_{t-1}(O_c^t) - l_0(O_c^t)
```

Donde:
- `l_t(O_c^t)`: log-odds de la belief actual
- `P(O_c^t | ξ_c^t)`: probabilidad de la CNN (nuestra `likelihood`)
- `l_{t-1}(O_c^t)`: log-odds de la belief anterior (warped)
- `l_0(O_c^t)`: log-odds del prior inicial (0 = neutral)

**Implementación correcta en `lidar_pipeline_suite.py` (líneas 1459-1491)**:

```python
# Convertir likelihood a probabilidad
prob_from_likelihood = 1.0 / (1.0 + np.exp(-likelihood_image))

# Convertir probabilidad a log-odds
prob_clamped = np.clip(prob_from_likelihood, 1e-6, 1.0 - 1e-6)
likelihood_log_odds = np.log(prob_clamped / (1.0 - prob_clamped))

# Prior inicial l_0: asumimos 0 (estado neutral)
l0 = 0.0

# Actualización Bayesiana (Eq. 9 de Dewan)
belief_map = likelihood_log_odds + belief_map_warped - l0

# Clamp para evitar saturación numérica
belief_map = np.clip(belief_map, -10, 10)
```

### 2. Comparación con `range_projection.py`

**Hallazgo clave**: Ambas implementaciones usan la misma ecuación de Dewan, pero difieren en el **tipo de probabilidad de entrada**.

**`range_projection.py` (líneas 998-1000, 1185-1218)**:
```python
def get_raw_probability(self, range_image):
    """Returns raw probability map derived directly from delta_r."""
    return (range_image < self.threshold_obs).astype(np.float32)  # BINARIA: 0.0 o 1.0

def update_belief(self, belief_map_state, P_obs, points_current):
    # P_obs es probabilidad BINARIA (0 o 1)
    warped_belief = self.warp_belief_map(belief_map_state, ...)

    P_clamped = np.clip(P_obs, 1e-6, 1.0 - 1e-6)
    l_obs = np.log(P_clamped / (1.0 - P_clamped))

    new_belief_map = warped_belief + l_obs - self.l0
    new_belief_map = np.clip(new_belief_map, -2.5, 2.5)

    return P_belief, new_belief_map
```

**Nuestra implementación**:
```python
# likelihood tiene valores CONTINUOS:
# -2.5 (ground), 1.5 (void), 2.0 (obstacle), 4.0 (HCD boost)

# Convierte likelihood continuo a probabilidad continua
prob_from_likelihood = 1.0 / (1.0 + np.exp(-likelihood_image))  # [0, 1]

# Luego aplica Bayes Filter con probabilidad continua
```

**Diferencia clave**:
- `range_projection.py`: **Probabilidad BINARIA** (0 o 1) → más agresiva, menos dilución
- `lidar_pipeline_suite.py`: **Probabilidad CONTINUA** (0-1) → valores "grises" pueden diluir la señal

### 3. Intentos de Solución

#### **Opción A: Calcular likelihood DIRECTAMENTE en range image**

**Motivación**: Evitar compresión lossy 20:1 (124k puntos → 6k pixels).

**Implementación** (líneas 1170-1275 en `lidar_pipeline_suite.py`):
```python
def compute_delta_r_on_range_image(self, range_image, local_planes_dict, ...):
    # Para cada pixel, calcular delta_r del punto más cercano
    # y asignar likelihood directamente al pixel
    ...
```

**Problema**: Al usar "closest point wins", obstacles detrás de ground en el mismo pixel se pierden.

**Resultados**:
- "Closest point wins": Recall 42.53% (igual que antes)
- "MAX likelihood wins": Recall 99.29%, Precision 20.08% (demasiado agresivo, 5x más FP)

**Conclusión**: NO resuelve el problema.

#### **Opción B: Volver a per-point likelihood + Bayes Filter de Dewan**

**Implementación** (líneas 1560-1603):
```python
def stage3_complete(self, points, delta_pose=None):
    # Stage 2: Delta-r + HCD Fusion (per-point)
    stage2_result = self.stage2_complete(points)

    # Proyectar likelihood a range image (closest point wins)
    range_proj = self.project_to_range_image(
        points=points,
        likelihood=stage2_result['likelihood']
    )

    # Stage 3: Bayes Filter (Dewan Eq. 9)
    stage3_result = self.update_belief_map(...)
    ...
```

**Resultados con 1 frame**:
- Recall: 43.09% (IGUAL que antes)
- Precision: 67.81%
- F1: 52.70%

**Resultados con 5 frames**:
- Recall: 42.97% (NO mejora con acumulación temporal)
- Precision: 69.86%
- F1: 53.21%

**Conclusión**: El Bayes Filter está bien implementado, pero NO resuelve el problema de recall.

---

## Diagnóstico del Problema

### Causa Raíz: Compresión Lossy 20:1

**Análisis de flujo**:

1. **Stage 2 (per-point)**:
   - Detecta 35,202 obstacles (de 124,231 puntos)
   - Recall: 91.6% ✓✓✓

2. **Proyección a range image**:
   - 124,231 puntos → 6,169 pixels únicos
   - Compresión: 20:1
   - "Closest point wins" (orden por range descendente)

3. **Resultado en range image**:
   - 3,998 pixels con likelihood > 1.0
   - Pérdida: 35,202 → 15,293 obstacles (56% perdidos)
   - Recall: 43.09% ✗✗✗

**Ejemplo del problema**:

```
Pixel (u=0, v=1343) contiene 5 puntos:
  - Punto 204 (obstacle): likelihood=2.0, range=24.95m
  - Punto 7411 (ground): likelihood=-2.5, range=21.57m  ← MÁS CERCANO, GANA
  - Punto 7412 (ground): likelihood=-2.5, range=21.57m
  - Punto 7413 (ground): likelihood=-2.5, range=21.90m
  - Punto 7414 (obstacle): likelihood=2.0, range=26.02m

Resultado: Pixel hereda likelihood=-2.5 del ground (punto más cercano)
→ Los 5 puntos (incluyendo 2 obstacles) se clasifican como ground
```

### ¿Por qué `range_projection.py` no tiene este problema?

**Hipótesis**: Usa probabilidad BINARIA en lugar de continua.

**Evidencia**:
```python
# range_projection.py línea 590
self.range_image[u_sorted, v_sorted] = delta_sorted  # Proyecta delta_r, NO range

# Luego línea 729-733
P_raw_2d = self.get_raw_probability(self.range_image)
# = (delta_r < -0.3).astype(np.float32)  # BINARIA: 0 o 1

P_belief, self.belief_map = self.update_belief(self.belief_map, P_raw_2d, points)
```

**Ventaja de probabilidad binaria**:
- Valores "grises" (likelihood ≈ 0) se eliminan
- Solo importa si es obstacle (P=1) o no (P=0)
- Más agresivo, menos dilución de señal

---

## Próximos Pasos

### Opción 1: Probar Probabilidad BINARIA (RECOMENDADO)

**Implementar en `lidar_pipeline_suite.py`**:

```python
# En lugar de usar likelihood continuo (-2.5, 1.5, 2.0, 4.0)
# Convertir a probabilidad binaria como range_projection.py

# Proyectar delta_r (NO likelihood) a range image
delta_r_image = np.zeros((H, W))
delta_r_image[u_sorted, v_sorted] = delta_r[valid_idx][order]

# Probabilidad binaria
P_binary = (delta_r_image < -0.3).astype(np.float32)  # 0.0 o 1.0

# Aplicar Bayes Filter con probabilidad binaria
P_belief, belief_map = update_belief(belief_map_warped, P_binary, ...)
```

**Predicción**: Recall mejorará porque:
- Evita dilución por valores intermedios
- Más similar a `range_projection.py` (que funciona bien)

### Opción 2: Aumentar Resolución de Range Image

**Cambiar de H=64 a H=256**:
- Reduce compresión de 20:1 a 5:1
- Requiere más memoria y cómputo
- Puede no resolver el problema fundamental

### Opción 3: No Usar Range Image para Stage 3

**Mantener per-point hasta el final**:
- Stage 2 per-point: 91.6% recall ✓
- Stage 3 per-point: ¿?
- Problema: Bayes Filter necesita estructura 2D para warping por egomotion

---

## Archivos Modificados

1. **`lidar_pipeline_suite.py`**:
   - Líneas 1170-1275: `compute_delta_r_on_range_image()` (Opción A - no funcionó)
   - Líneas 1459-1491: `update_belief_map()` con Bayes Filter de Dewan ✓
   - Líneas 1560-1603: `stage3_complete()` refactorizado

2. **Tests creados**:
   - `tests/test_range_projection_standalone.py`: Intento de test standalone (incompleto)

---

## Referencias

- **Paper**: "DeepTemporalSeg: Temporally Consistent Semantic Segmentation of 3D LiDAR Scans" (Dewan & Burgard, IROS 2018)
- **Ecuación clave**: Ecuación 9 (Bayes Filter con log-odds)
- **Código referencia**: `range_projection.py` líneas 998-1000, 1185-1218
- **Documentación**: `ALGORITMO_OPTIMO_DETECCION_OBSTACULOS.md`

---

## Estado Actual

### Implementación Correcta ✓
- Bayes Filter de Dewan (Eq. 9) implementado correctamente
- Warping de egomotion funciona (depth-jump check)
- Proyección a range image usa "closest point wins"

### Problema Pendiente ✗
- **Recall 43% vs 92% esperado** (-49 puntos)
- Compresión 20:1 pierde 56% de obstacles
- Probabilidad continua puede estar diluyendo señal

### Próxima Acción
**PROBAR OPCIÓN 1**: Probabilidad binaria como `range_projection.py`

---

## Comando para Retomar

```bash
cd /home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea

# Test actual (recall 43%)
python3 tests/test_stage3_bayesian_filter.py --scan_start 0 --n_frames 5

# Ver implementación de probabilidad binaria en range_projection.py
grep -n "get_raw_probability\|P_raw_2d" range_projection.py -A 5
```

---

**Nota**: Este documento resume 2 horas de trabajo analizando papers, código y debugging. El hallazgo clave es que `range_projection.py` usa **probabilidad BINARIA** mientras nosotros usamos **probabilidad CONTINUA**, lo cual puede explicar la diferencia en recall.
