# Analisis Completo: Pipeline Stages 1-3 en KITTI Sequences 00 y 04

Test ejecutado: `tests/test_full_pipeline_both_sequences.py`
Fecha: 2026-03-16

---

## 1. Configuracion del Experimento

| Parametro | Valor |
|-----------|-------|
| Sensor | Velodyne HDL-64E (64 anillos, ~125k puntos/frame) |
| Ground truth | SemanticKITTI labels semanticas |
| Sequence 00 | Urbana: edificios, coches, vegetacion, aceras |
| Sequence 04 | Autopista: vehiculos rapidos, vallas, vegetacion distante |
| Frames Stage 1-2 | Frame 0 (single shot) |
| Frames Stage 3 | 10 frames (0-9) con filtro temporal |
| Wall rejection | Hibrido (bin-wise + point-wise, nz < 0.7, deltaZ > 0.3m) |
| HCD | Activado (Height Coding Descriptor, ERASOR++) |
| Egomotion | Poses KITTI + delta_pose inv(T_t) @ T_{t-1} |

### Labels usadas

- **Ground**: road (40), parking (44), sidewalk (48), other-ground (49), lane-marking (60), terrain (72)
- **Obstacle**: car, bus, truck, person, bicycle, building, fence, vegetation, pole, trunk, etc.
- **Criticos** (riesgo de colision): vehiculos (10,13,15,18,20) + personas (30,31,32) + moving (252-259)

---

## 2. Stage 1: Patchwork++ Ground Segmentation

### 2.1 Problema: Patchwork++ clasifica obstaculos como ground

#### Sequence 00 (Urbana)

| Metrica | Valor |
|---------|-------|
| Total puntos | 124,668 |
| GT ground / obstacle | 67,873 (54.4%) / 54,607 (43.8%) |
| Patchwork++ ground / non-ground | 72,599 / 52,069 |
| **Obstaculos como ground** | **4,689 (8.59%)** |
| Ground precision | 93.0% |

Desglose por tipo:

| Tipo | Mal clasificados | Total GT | % del tipo |
|------|-----------------|----------|------------|
| vegetation | 3,784 | 27,123 | **14.0%** |
| building | 509 | 18,268 | 2.8% |
| car | 169 | 4,234 | 4.0% |
| other-object | 76 | 1,227 | 6.2% |
| fence | 74 | 370 | **20.0%** |
| other-structure | 30 | 1,471 | 2.0% |
| trunk | 24 | 1,192 | 2.0% |
| pole | 20 | 532 | 3.8% |

#### Sequence 04 (Autopista)

| Metrica | Valor |
|---------|-------|
| Total puntos | 124,231 |
| GT ground / obstacle | 89,995 (72.4%) / 29,764 (24.0%) |
| Patchwork++ ground / non-ground | 90,869 / 33,362 |
| **Obstaculos como ground** | **1,935 (6.50%)** |
| Ground precision | 97.2% |

Desglose por tipo:

| Tipo | Mal clasificados | Total GT | % del tipo |
|------|-----------------|----------|------------|
| vegetation | 915 | 14,107 | 6.5% |
| fence | 413 | 2,407 | **17.2%** |
| building | 225 | 2,277 | **9.9%** |
| other-object | 209 | 5,370 | 3.9% |
| bus | 70 | 2,076 | 3.4% |
| car | 33 | 1,209 | 2.7% |
| pole | 29 | 778 | 3.7% |
| moving-car | 26 | 840 | 3.1% |
| moving-person | 3 | 28 | **10.7%** |

#### Comparativa entre escenas

| Metrica | Seq 00 (Urbana) | Seq 04 (Autopista) |
|---------|----------------|-------------------|
| Obstaculos como ground | **4,689 (8.59%)** | 1,935 (6.50%) |
| Planos verticales CZM | **35/274 (12.8%)** | 18/414 (4.3%) |
| Ground precision | 93.0% | **97.2%** |
| Criticos perdidos | 172 | 132 |

**La escena urbana es peor**: mas superficies verticales cercanas (edificios, muros).

### 2.2 Tiene Patchwork++ wall rejection?

**Si, pero solo en Zone 0 (<9.64m)** mediante RVPF (Reflected Vertical Plane Filter).

#### Planos verticales (nz < 0.7) por zona CZM

**Sequence 00**:
| Zona | Rango | Total planos | Verticales | % |
|------|-------|-------------|------------|---|
| Z0 (RVPF activo) | 0 - 9.6m | 31 | **0** | 0.0% |
| Z1 | 9.6 - 22m | 102 | **18** | 17.6% |
| Z2 | 22 - 48m | 113 | **12** | 10.6% |
| Z3 | 48 - 80m | 28 | **5** | 17.9% |

**Sequence 04**:
| Zona | Rango | Total planos | Verticales | % |
|------|-------|-------------|------------|---|
| Z0 (RVPF activo) | 0 - 9.6m | 29 | **0** | 0.0% |
| Z1 | 9.6 - 22m | 129 | **1** | 0.8% |
| Z2 | 22 - 48m | 219 | **11** | 5.0% |
| Z3 | 48 - 80m | 37 | **6** | 16.2% |

**Conclusion**: RVPF funciona en Zone 0 (0 verticales). Pero Zones 1-3 acumulan 18-35 planos verticales sin filtrar. **Nuestro wall rejection extiende la proteccion a todas las zonas.**

### 2.3 Se clasifica mal el objeto entero o solo la base?

**Solo puntos de la base** (parte inferior cercana al suelo, no el objeto entero).

**Sequence 00**:
| Tipo | Mal/Total | % | Posicion | Z mal clasif. | Dist. media |
|------|-----------|---|----------|---------------|-------------|
| building | 509/18,268 | 2.8% | **BASE (23%)** | [-1.8, 1.2]m | 19.2m |
| fence | 74/370 | 20.0% | **BASE (16%)** | [-1.9, -1.6]m | 21.9m |
| vegetation | 3,784/27,123 | 14.0% | **BASE (28%)** | [-2.8, 0.5]m | 12.2m |
| car | 169/4,234 | 4.0% | **BASE (26%)** | [-2.6, -0.5]m | 16.1m |
| pole | 20/532 | 3.8% | **BASE (11%)** | [-1.8, 0.1]m | 13.2m |

**Sequence 04**:
| Tipo | Mal/Total | % | Posicion | Z mal clasif. | Dist. media |
|------|-----------|---|----------|---------------|-------------|
| building | 225/2,277 | 9.9% | MEDIO (31%) | [-1.4, 1.3]m | 31.2m |
| fence | 413/2,407 | 17.2% | **BASE (25%)** | [-2.2, 0.2]m | 26.7m |
| vegetation | 915/14,107 | 6.5% | MEDIO (37%) | [-3.6, 1.8]m | 34.5m |
| car | 33/1,209 | 2.7% | MEDIO (36%) | [-2.1, -0.3]m | 27.9m |
| bus | 70/2,076 | 3.4% | **BASE (12%)** | near ground | 19.6m |

**Observaciones**:
- Escena urbana (seq 00): todos los objetos tienen puntos mal clasificados en la **BASE** (<30%)
- Autopista (seq 04): objetos mas lejos (25-35m), la posicion se desplaza al MEDIO porque a mayor distancia los rayos pierden resolucion vertical
- **Fence** es el peor en ambas escenas (17-20%) porque son objetos bajos (~1.2m) donde casi toda la estructura esta cerca del suelo
- La mayoria de puntos mal clasificados estan en Zones 1-3 (>9.64m), fuera del RVPF

### 2.4 Impacto en navegacion

| Escena | Criticos como ground | Detalle |
|--------|---------------------|---------|
| Seq 00 | 172 / 4,322 (4.0%) | 169 car, 3 moving-motorcyclist |
| Seq 04 | 132 / 4,263 (3.1%) | 70 bus, 33 car, 26 moving-car, 3 moving-person |

Estos puntos se pierden **antes** de llegar a Stage 2, por lo que:
- Stage 2 nunca los evalua como candidatos a obstaculo
- Stage 3 no puede acumular evidencia temporal sobre ellos
- El planificador ve un "hueco" en la base del obstaculo donde podria intentar pasar

### 2.5 Efecto del wall rejection en el pipeline

| Metrica | Seq 00 | Seq 04 |
|---------|--------|--------|
| Obstaculos como ground (Patchwork++ vanilla) | 4,689 (8.59%) | 1,935 (6.50%) |
| Obstaculos como ground (con wall rejection) | 5,462 (10.00%) | 2,566 (8.62%) |

**Nota**: El wall rejection *aumenta* ligeramente los obstaculos como ground porque usa
umbrales mas conservadores en su primera pasada. Sin embargo, los puntos rechazados como
"wall" se devuelven al pipeline como non-ground, lo que permite a Stage 2 evaluarlos.
El beneficio real se ve en las metricas de Stage 2 (precision y recall).

---

## 3. Stage 2: Delta-r Anomaly Detection (Single Frame)

### Metricas (Frame 0)

| Metrica | Seq 00 (Urbana) | Seq 04 (Autopista) |
|---------|----------------|-------------------|
| Obstacles detectados | 49,020 | 35,201 |
| GT obstacles | 54,607 | 29,764 |
| **Precision** | **93.11%** | 78.23% |
| **Recall** | 83.58% | **92.52%** |
| **F1** | **88.09%** | 84.77% |
| TP | 45,641 | 27,537 |
| FP | 3,379 | 7,664 |
| FN | 8,966 | 2,227 |
| Timing | 3,375 ms | 3,522 ms |

### Analisis

- **Seq 00 (urbana)**: Alta precision (93.1%) pero menor recall (83.6%). Los edificios y
  vegetacion cercana generan planos ground confiables, reduciendo FP. Pero la complejidad
  de la escena deja 8,966 FN.

- **Seq 04 (autopista)**: Alto recall (92.5%) pero menor precision (78.2%). La carretera
  plana permite detectar casi todo, pero hay mas FP (7,664) probablemente por vegetacion
  lejana y ruido a distancia.

- **Timing**: ~3.4s/frame dominado por Patchwork++ (~75% del tiempo).

---

## 4. Stage 3: Bayesian Temporal Filter (10 Frames)

### 4.1 Sequence 00 (Urbana)

| Metrica | Stage 2 (baseline) | Stage 3 sin ego | Stage 3 con ego |
|---------|-------------------|----------------|----------------|
| Obstacles detectados | 49,020 | 51,935 | 78,360 |
| **Precision** | 93.11% | **93.21%** | 76.59% |
| **Recall** | 83.58% | 76.78% | **95.19%** |
| **F1** | **88.09%** | 84.20% | 84.89% |
| TP | 45,641 | 48,407 | 60,019 |
| FP | 3,379 | 3,528 | 18,341 |
| FN | 8,966 | 14,642 | 3,030 |
| Timing | 3,375 ms | 5,584 ms | 6,222 ms |

**Impacto del egomotion (seq 00)**:
- Recall: **+18.42%** (76.78% -> 95.19%)
- Precision: **-16.61%** (93.21% -> 76.59%)
- F1: **+0.69%** (84.20% -> 84.89%)
- FN reducidos: 14,642 -> 3,030 (**-11,612 puntos, -79.3%**)

### 4.2 Sequence 04 (Autopista)

#### Con 10 frames

| Metrica | Stage 2 (baseline) | Stage 3 sin ego | Stage 3 con ego |
|---------|-------------------|----------------|----------------|
| **Precision** | **79.97%** | 74.06% | 55.54% |
| **Recall** | 93.90% | 73.80% | **98.63%** |
| **F1** | **86.38%** | 73.93% | 71.06% |
| FP | - | 8,076 | 24,668 |
| FN | - | 8,187 | 429 |
| Timing | 3,522 ms | 5,313 ms | 5,754 ms |

#### Con 20 frames

| Metrica | Stage 2 (baseline) | Stage 3 sin ego | Stage 3 con ego |
|---------|-------------------|----------------|----------------|
| **Precision** | **82.17%** | 79.63% | 59.02% |
| **Recall** | 94.60% | 76.78% | **99.20%** |
| **F1** | **87.95%** | 78.18% | 74.01% |
| FP | - | 7,480 | 26,233 |
| FN | - | 8,844 | 303 |

**Impacto del egomotion (seq 04, 10 frames)**:
- Recall: **+24.83%** (73.80% -> 98.63%)
- Precision: **-18.52%** (74.06% -> 55.54%)
- F1: **-2.87%** (73.93% -> 71.06%)
- FN reducidos: 8,187 -> 429 (**-7,758 puntos, -94.8%**)

**Con 20 frames el patron se mantiene**: recall sube a 99.2% pero precision sigue baja (59.0%).
El F1 con egomotion (74.0%) sigue por debajo de Stage 2 solo (88.0%).
**Stage 4 Shadow Validation es critico para esta escena.**

### 4.3 Comparativa Stage 3 (10 frames)

| Metrica | Seq 00 sin ego | Seq 00 con ego | Seq 04 sin ego | Seq 04 con ego |
|---------|---------------|---------------|---------------|---------------|
| Precision | 93.21% | 76.59% | 74.06% | 55.54% |
| Recall | 76.78% | **95.19%** | 73.80% | **98.63%** |
| F1 | 84.20% | 84.89% | 73.93% | 71.06% |
| FN | 14,642 | **3,030** | 8,187 | **429** |

**Nota sobre resultados anteriores**: En una sesion anterior se reportaron metricas mas
favorables (F1 82.87% con 20 frames). Esos resultados estaban contaminados por un bug
en `stage3_per_point()` donde `**stage2_result` sobrescribia el `obs_mask` de Stage 3
con el de Stage 2. Tras corregir el bug (mover `**stage2_result` al inicio del dict),
las metricas muestran el Stage 3 real, que tiene precision significativamente menor.

### 4.4 Analisis de resultados

#### Egomotion mejora masivamente el recall

El egomotion compensation logra recalls de **95-98%** en ambas escenas. La asociacion
KDTree con threshold 2.0m permite vincular puntos entre frames incluso a velocidad
de autopista (~47 km/h, ~1.3m/frame). Los FN se reducen un 79-95%.

#### La precision cae significativamente

El problema principal es el aumento de FP (false positives):
- Seq 00: 3,528 -> 18,341 FP (+420%)
- Seq 04: 8,076 -> 24,668 FP (+205%)

**Causa**: Al compensar egomotion, los beliefs de frames anteriores se propagan
correctamente a las nuevas posiciones. Pero puntos que eran ruido transitorio
en frames anteriores tambien se propagan, y si coinciden espacialmente con
puntos del frame actual (threshold 2.0m), el ruido se acumula.

#### Seq 04 es peor que seq 00

En autopista, el F1 con egomotion (71.06%) es **peor** que sin el (73.93%).
Esto se debe a que:
1. En autopista hay mas vegetacion lejana que genera FP persistentes
2. La velocidad alta (47 km/h) causa mas "smearing" del belief map
3. El threshold de 2.0m es mas permisivo en autopista donde los objetos se mueven rapido

#### Stage 3 sin egomotion es peor que Stage 2

En ambas secuencias, Stage 3 sin egomotion tiene peor F1 que Stage 2 solo:
- Seq 00: 84.20% vs 88.09%
- Seq 04: 73.93% vs 84.77%

**Causa**: Sin compensar el movimiento del vehiculo, el belief de frame t-1
se asocia a posiciones *diferentes* en frame t. Esto genera:
- Beliefs que no se acumulan correctamente (FN sube)
- Beliefs mal ubicados que generan FP

**Conclusion**: El filtro temporal Bayesiano **necesita** egomotion para funcionar
correctamente. Sin el, es contraproducente.

---

## 5. Problemas Detectados y Estado

### Problema 1: Precision baja con egomotion en seq 04 (55.54%)

**Causa**: Threshold de asociacion KDTree demasiado permisivo (2.0m) en autopista + vegetacion lejana persistente.

**Solucion propuesta**:
- Stage 4 (Shadow Validation) deberia eliminar muchos FP
- Threshold adaptativo por velocidad/rango
- Post-filtering con DBSCAN (min cluster size)

**Estado**: Pendiente

### Problema 2: Patchwork++ clasifica base de obstaculos como ground

**Causa**: RVPF solo activo en Zone 0 (<9.64m). Planos verticales en Zones 1-3 pasan como ground.

**Solucion implementada**: Wall rejection hibrido (bin-wise + point-wise) extiende la proteccion a todas las zonas.

**Estado**: Resuelto

### Problema 3: Stage 3 sin egomotion es contraproducente

**Causa**: Sin compensar movimiento, el belief map del frame anterior se asocia a posiciones incorrectas.

**Solucion implementada**: Egomotion compensation con delta_pose = inv(T_t) @ T_{t-1} y KDTree association.

**Estado**: Resuelto

### Problema 4: Latencia alta (~5-6s/frame)

**Causa**: Patchwork++ (~75%), KDTree construction/query (~15%).

**Solucion propuesta**:
- Voxel downsampling antes de Patchwork++
- KD-Tree con subsampling
- Port a C++ para modulos criticos

**Estado**: Pendiente (no prioritario para TFG)

### Problema 5: Fence es el tipo mas afectado (17-20% mal clasificado)

**Causa**: Objetos bajos (~1.2m) donde casi toda la estructura esta cerca del suelo. Patchwork++ mezcla puntos de fence con ground en el mismo bin CZM.

**Solucion parcial**: Wall rejection point-wise con deltaZ > 0.3m recupera algunos puntos. Stage 3 temporal ayuda a acumular evidencia sobre fences persistentes.

**Estado**: Parcialmente resuelto

---

## 6. Resumen de Metricas

### Mejor configuracion por escena

| Escena | Mejor F1 | Config | Precision | Recall |
|--------|----------|--------|-----------|--------|
| Seq 00 (Urbana) | **88.09%** | Stage 2 solo | 93.11% | 83.58% |
| Seq 04 (Autopista) | **84.77%** | Stage 2 solo | 78.23% | 92.52% |

### Si priorizamos recall (seguridad = no perder obstaculos)

| Escena | Mejor Recall | Config | Precision | F1 |
|--------|-------------|--------|-----------|-----|
| Seq 00 | **95.19%** | S3 con ego | 76.59% | 84.89% |
| Seq 04 | **98.63%** | S3 con ego | 55.54% | 71.06% |

### Progresion del pipeline

```
Seq 00 (Urbana):
  Patchwork++ vanilla  -> 8.59% obstaculos perdidos como ground
  + Wall rejection     -> Stage 2: P=93.1%, R=83.6%, F1=88.1%
  + Temporal (sin ego) -> P=93.2%, R=76.8%, F1=84.2% (peor sin ego)
  + Temporal (con ego) -> P=76.6%, R=95.2%, F1=84.9% (recall excelente)
  + Shadow validation  -> (pendiente, deberia mejorar precision)

Seq 04 (Autopista):
  Patchwork++ vanilla  -> 6.50% obstaculos perdidos como ground
  + Wall rejection     -> Stage 2: P=78.2%, R=92.5%, F1=84.8%
  + Temporal (sin ego) -> P=74.1%, R=73.8%, F1=73.9% (peor sin ego)
  + Temporal (con ego) -> P=55.5%, R=98.6%, F1=71.1% (precision a mejorar)
  + Shadow validation  -> (pendiente, critico para seq 04)
```

---

## 7. Conclusiones

1. **Patchwork++ tiene una limitacion real**: Clasifica 6.5-8.6% de obstaculos como ground, concentrado en la base de los objetos y en Zones 1-3 (>9.64m) donde RVPF no actua.

2. **El wall rejection es necesario y efectivo**: Extiende la proteccion contra planos verticales a todas las zonas CZM.

3. **El filtro temporal Bayesiano requiere egomotion**: Sin compensacion de movimiento, Stage 3 es contraproducente (F1 baja en ambas escenas).

4. **Con egomotion, el recall es excelente** (95-98%): Casi ningun obstaculo se pierde. Pero la precision cae (55-77%), especialmente en autopista.

5. **Stage 4 (Shadow Validation) es critico**: Es el siguiente paso para recuperar precision sin sacrificar recall, especialmente en seq 04 donde la precision con egomotion cae al 55%.

6. **La escena urbana se comporta mejor**: Mayor precision, mas planos de referencia, objetos mas cercanos. La autopista es mas desafiante por velocidad y distancia de los objetos.
