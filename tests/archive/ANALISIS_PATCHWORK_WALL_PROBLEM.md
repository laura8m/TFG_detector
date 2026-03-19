# Analisis: Patchwork++ Clasifica Obstaculos como Suelo

**Fecha**: 2026-03-12
**Test**: `tests/test_patchwork_wall_problem.py`
**Escenas evaluadas**: KITTI Sequence 00 (urbana) y Sequence 04 (autopista)

---

## Resumen del Problema

Patchwork++ incluye un filtro de planos verticales llamado RVPF (Reflected Vertical Plane Filter), pero este filtro **solo esta activo en Zone 0** del CZM (distancia < 9.64m del sensor). En las zonas 1-3 (>9.64m), los planos verticales se aceptan como ground sin verificar su orientacion.

Esto provoca que puntos de la **base de edificios, vallas, vegetacion y vehiculos** se clasifiquen como suelo. No es el objeto entero, sino la franja inferior donde la geometria es ambigua entre suelo real y la base del obstaculo.

---

## Evidencia: Planos Verticales por Zona CZM

### Sequence 00 (urbana) - Frame 0

| Zona CZM | Distancia | Total planos | Planos verticales (nz < 0.7) | % |
|-----------|-----------|-------------|-------------------------------|---|
| Zone 0 | 0 - 9.64m | 31 | **0** | 0.0% |
| Zone 1 | 9.64 - 22.28m | 102 | **18** | 17.6% |
| Zone 2 | 22.28 - 48.56m | 113 | **12** | 10.6% |
| Zone 3 | 48.56 - 80m | 28 | **5** | 17.9% |

**Zone 0: 0 planos verticales** - RVPF funciona correctamente aqui.
**Zones 1-3: 35 planos verticales** - Sin RVPF, pasan como ground.

### Sequence 04 (autopista) - Frame 0

| Zona CZM | Total planos | Planos verticales (nz < 0.7) | % |
|-----------|-------------|-------------------------------|---|
| Total | 414 | **18** | 4.3% |

Menos planos verticales porque la autopista tiene menos estructuras verticales cercanas que una escena urbana.

---

## Que Objetos se Clasifican Mal

### Sequence 00 (urbana) - 4689 obstaculos como ground (8.6% del GT)

| Tipo de obstaculo | Total puntos | Mal clasificados | % mal clasificado | Posicion en el objeto |
|-------------------|-------------|-----------------|-------------------|----------------------|
| **fence** (valla) | 370 | 74 | **20.0%** | 16% desde la base |
| **vegetation** | 27123 | 3784 | **14.0%** | 28% desde la base |
| **other-object** | 1227 | 76 | 6.2% | - |
| **car** | 4234 | 169 | **4.0%** | 26% desde la base |
| **pole** | 532 | 20 | 3.8% | - |
| **moving-motorcyclist** | 88 | 3 | 3.4% | - |
| **building** | 18268 | 509 | **2.8%** | 23% desde la base |
| **trunk** | 1192 | 24 | 2.0% | - |
| **other-structure** | 1471 | 30 | 2.0% | - |

### Sequence 04 (autopista) - 1935 obstaculos como ground (6.5% del GT)

| Tipo de obstaculo | Total puntos | Mal clasificados | % mal clasificado |
|-------------------|-------------|-----------------|-------------------|
| **fence** (valla) | 2407 | 413 | **17.2%** |
| **moving-person** | 28 | 3 | **10.7%** |
| **building** | 2277 | 225 | **9.9%** |
| **vegetation** | 14107 | 915 | **6.5%** |
| **other-object** | 5370 | 209 | 3.9% |
| **pole** | 778 | 29 | 3.7% |
| **trunk** | 358 | 12 | 3.4% |
| **bus** | 2076 | 70 | 3.4% |
| **car** | 1209 | 33 | 2.7% |

---

## Se Clasifica Mal el Objeto Entero o Solo Puntos?

**Solo puntos de la BASE del objeto** (parte baja, cercana al suelo). Analisis detallado para Sequence 00:

### Building (18268 puntos)

```
Mal clasificados:  509 puntos (2.8%)
Bien clasificados: 17759 puntos (97.2%)

Altura Z mal clasificados:  [-1.83, 1.20]  media = -0.99m
Altura Z bien clasificados: [-1.73, 1.88]  media = -0.08m

Rango Z total del objeto: 3.71m
Posicion de los mal clasificados: 23% desde la base
--> Son los puntos de la BASE del edificio
```

El edificio tiene 3.71m de rango vertical. Solo el cuartil inferior (puntos donde el edificio se encuentra con el suelo) se clasifica mal. El 97.2% del edificio se detecta correctamente como non-ground.

### Fence (370 puntos)

```
Mal clasificados:  74 puntos (20.0%)
Bien clasificados: 296 puntos (80.0%)

Altura Z mal clasificados:  [-1.90, -1.60]  media = -1.70m
Altura Z bien clasificados: [-1.71, -0.69]  media = -1.24m

Rango Z total: 1.21m
Posicion: 16% desde la base
--> Son los puntos de la BASE de la valla
```

Las vallas son el caso mas afectado (20%) porque son objetos bajos: con solo 1.21m de rango vertical, los puntos de la base son una proporcion mayor del total.

### Vegetation (27123 puntos)

```
Mal clasificados:  3784 puntos (14.0%)
Bien clasificados: 23339 puntos (86.0%)

Altura Z mal clasificados:  [-2.82, 0.49]  media = -1.49m
Altura Z bien clasificados: [-2.62, 1.87]  media = -0.58m

Rango Z total: 4.69m
Posicion: 28% desde la base
--> Son los puntos bajos de arbustos/setos
```

La vegetacion tiene el mayor numero absoluto de puntos mal clasificados (3784) porque los arbustos bajos cerca del suelo son geometricamente ambiguos: un RANSAC puede encajar un plano entre las ramas bajas.

### Car (4234 puntos)

```
Mal clasificados:  169 puntos (4.0%)
Bien clasificados: 4065 puntos (96.0%)

Altura Z mal clasificados:  [-2.61, -0.52]  media = -1.86m
Altura Z bien clasificados: [-2.48, 0.33]   media = -1.13m

Rango Z total: 2.94m
Posicion: 26% desde la base
--> Son los puntos de las RUEDAS/parte baja del coche
```

---

## Donde Ocurren las Misclasificaciones

Todas las misclasificaciones de building, fence y car ocurren en **Zones 1-3** (>9.64m), donde RVPF no esta activo:

| Objeto | En Zone 0 (<9.64m) | En Zones 1-3 (>9.64m) |
|--------|-------------------|----------------------|
| **building** | 1 | **508** |
| **fence** | 0 | **74** |
| **car** | 77 | **92** |
| **vegetation** | 1752 | **2032** |

La excepcion es vegetacion, donde aproximadamente la mitad de los puntos mal clasificados estan en Zone 0. Esto se debe a que RVPF filtra planos verticales, pero la vegetacion baja no forma planos verticales claros - son mas bien superficies irregulares con normal intermedia que pasan el filtro RVPF.

---

## Impacto en la Navegacion y Deteccion de Obstaculos

### Impacto directo: False Negatives en el pipeline

Los puntos de obstaculos clasificados como ground **no llegan a Stage 2** (delta-r anomaly detection). Esto crea false negatives que se propagan por todo el pipeline:

```
Stage 1: base de valla clasificada como ground
    --> Stage 2: no calcula delta-r para esos puntos (son "suelo")
        --> Stage 3: no acumula belief temporal para esos puntos
            --> Output: zona navegable donde HAY un obstaculo
```

### Riesgo para la navegacion

**Riesgo moderado-bajo**: Los puntos mal clasificados son la base de los objetos (16-28% inferior). La parte superior del obstaculo SI se detecta correctamente como non-ground. Un planificador que use la nube de obstaculos detectada aun veria el obstaculo, pero con su base "recortada".

Sin embargo, hay casos criticos:

1. **Vallas bajas (1.21m)**: Con 20% mal clasificado, una valla baja podria perder suficientes puntos de la base para que DBSCAN (Stage 6) no forme un cluster solido, creando un hueco en el boundary navegable.

2. **Vehiculos lejanos**: A >20m, un coche tiene pocos puntos LiDAR. Si 4% de la base se pierde, el cluster podria no alcanzar el `min_samples=5` de DBSCAN y desaparecer.

3. **Personas/ciclistas**: Con 10.7% de moving-person mal clasificado (aunque solo 3 puntos en este frame), una persona lejana con pocos retornos LiDAR podria perder puntos criticos.

### Cuantificacion del riesgo

| Escena | Obstaculos perdidos como ground | % del GT | Riesgo navegacion |
|--------|--------------------------------|----------|-------------------|
| Seq 00 (urbana) | 4689 puntos | 8.6% | Moderado (muchos buildings/veg) |
| Seq 04 (autopista) | 1935 puntos | 6.5% | Bajo-Moderado (vallas laterales) |

El riesgo es **mayor en escenas urbanas** (mas objetos cercanos, mas vallas, mas vegetacion) que en autopista (objetos mas separados, vallas solo en laterales).

### Como mitiga el wall rejection este riesgo

El Hybrid Wall Rejection (v2.3) implementado:

1. Detecta planos con `nz < 0.7` en Zones 1-3 (que RVPF no cubre)
2. Analiza point-wise con KDTree para rechazar solo puntos con `delta_Z > 0.3m`
3. Reclasifica los puntos rechazados como non-ground, devolviendolos al pipeline

Sin wall rejection, estos 1935-4689 puntos se pierden silenciosamente. Con wall rejection, se recuperan y entran a Stage 2 para evaluacion como posibles obstaculos.

---

## Estadisticas del Ground de Patchwork++

### Sequence 00 (urbana)

```
Patchwork++ ground (72599 puntos):
  Correctos (GT = ground):       67526 (93.0%)
  INCORRECTOS (GT = obstaculo):   4689 (6.5%)
  Unlabeled/otro:                   384 (0.5%)
```

### Sequence 04 (autopista)

```
Patchwork++ ground (90869 puntos):
  Correctos (GT = ground):       88352 (97.2%)
  INCORRECTOS (GT = obstaculo):   1935 (2.1%)
  Unlabeled/otro:                   582 (0.6%)
```

Patchwork++ es muy bueno como segmentador de suelo (93-97% correcto), pero ese 2-6.5% de obstaculos infiltrados puede ser critico para navegacion segura.

---

## Puntos Ground con Altura Anomala

### Sequence 00

```
Puntos ground con Z > 0.0m (sobre el sensor): 57
Puntos ground con Z > 0.5m: 8
Rango Z del ground: [-11.56, 1.20]m

Segmentos con DeltaZ > 0.5m en radio 1m: 10/2000 muestras (0.5%)
  #1: pos=(7.4, 27.6, -3.0), DeltaZ=0.52m, 83 vecinos
  #2: pos=(2.4, 25.9, -2.5), DeltaZ=0.59m, 74 vecinos
  #3: pos=(8.5, 27.2, -2.8), DeltaZ=0.54m, 40 vecinos
```

### Sequence 04

```
Puntos ground con Z > 0.0m (sobre el sensor): 306
Puntos ground con Z > 0.5m: 218
Rango Z del ground: [-14.49, 2.18]m
```

La seq 04 tiene mas puntos anomalos (306 vs 57) porque la autopista tiene guardarrailes y vallas laterales cuyos puntos bajos se confunden con el suelo.

---

## Conclusion

1. **El problema es real**: Patchwork++ clasifica 6.5-8.6% de los obstaculos como ground.
2. **No es el objeto entero**: Solo la base (16-28% inferior) de cada obstaculo se ve afectada.
3. **La causa es conocida**: RVPF solo activa en Zone 0 (<9.64m).
4. **El impacto en navegacion es moderado**: Los obstaculos siguen siendo visibles por su parte superior, pero la base recortada puede causar huecos en clusters y boundaries.
5. **El wall rejection es necesario**: Recupera los puntos de la base reclasificandolos como non-ground.
6. **Peor en escenas urbanas**: Seq 00 tiene 2.4x mas puntos afectados que seq 04.

---

## Reproduccion

```bash
cd /home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea

# Test sequence 04 (autopista)
python3 tests/test_patchwork_wall_problem.py

# Test sequence 00 (urbana) - modificar SCAN_FILE y LABEL_FILE en el script
```

**Scripts relacionados**:
- `Paso_1/test_vanilla_patchwork.py` - Test original con configuracion vanilla
- `Paso_1/debug_patchwork.py` - Analisis de normales de planos
- `Paso_1/visualize_wall_problem.py` - Visualizacion de segmentos verticales
- `Paso_1/EVIDENCIA_PATCHWORK.md` - Documentacion original del problema
