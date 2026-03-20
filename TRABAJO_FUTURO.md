# Trabajo Futuro

Extensiones posibles del pipeline actual (Stage 1-3) que no se han implementado pero se han evaluado conceptualmente.

---

## 1. Multi-Object Tracking (AB3DMOT)

**Paper**: [A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics](https://www.ri.cmu.edu/publications/ab3dmot-a-baseline-for-3d-multi-object-tracking-and-new-evaluation-metrics/) (Weng et al., IROS 2020)

**Que aporta**: Asocia detecciones entre frames consecutivos para mantener identidad de objetos. Permite filtrar FP transitorios (aparecen 1 frame y desaparecen) y recuperar FN por oclusion temporal.

**Como integrarlo**:
1. Extraer AABBs (Axis-Aligned Bounding Boxes) de los clusters DBSCAN del Stage 3
2. Implementar Kalman filter para predecir posicion del objeto en el siguiente frame
3. Hungarian matching para asociar predicciones con nuevas detecciones
4. Filtrar objetos que no persisten >= N frames (elimina FP esporadicos)

**Requisitos**: Poses del vehiculo (KITTI las proporciona en `poses.txt`). No requiere GPU ni entrenamiento — Kalman + Hungarian son algoritmos geometricos puros.

**Impacto estimado**: +1-3% F1 por eliminacion de FP transitorios y recuperacion de FN por oclusion.

---

## 2. Prediccion de Posicion con Odometria

**Que aporta**: Propaga la posicion estimada de objetos entre frames usando un modelo de movimiento. Permite anticipar donde estara un objeto ocluido temporalmente.

**Como integrarlo**:
1. Viene integrado con el tracking (Kalman filter ya predice posicion)
2. Usar poses de odometria de KITTI para compensar egomotion del vehiculo
3. El estado del Kalman filter incluye posicion + velocidad del objeto

**Requisitos**: Tracking implementado (punto 1) + poses de odometria. Sin tracking no hay identidad de objetos para propagar.

**Impacto estimado**: Marginal sobre tracking solo. Mejora principal en escenarios de oclusion prolongada (>3 frames).

---

## 3. Oriented Bounding Boxes + Clasificacion Semantica por Geometria

**Que aporta**: Detectar tipo de objeto (coche, peaton, ciclista) y su orientacion usando solo la geometria del cluster, sin deep learning.

**Como integrarlo**:
1. Oriented BBox: PCA sobre puntos del cluster → ejes principales → min-area rectangle 3D
2. Clasificacion heuristica por dimensiones del cluster:
   - Coche: ~4.0 x 1.8 x 1.5m, aspect ratio ~2.2
   - Peaton: ~0.5 x 0.5 x 1.7m, aspect ratio ~0.3
   - Ciclista: ~1.8 x 0.6 x 1.7m, aspect ratio ~3.0
3. Score de confianza basado en ajuste a dimensiones tipicas

**Requisitos**: Clusters del Stage 3. No requiere GPU ni entrenamiento.

**Limitaciones**: Clasificacion heuristica tiene baja fiabilidad en clusters parcialmente ocluidos o con pocos puntos. No comparable con metodos deep learning (~97% AP). Las metricas actuales (F1 binario obs/no-obs) no cambiarian — requeriria evaluar con metricas por clase (AP, como KITTI benchmark).

---

## 4. Clasificacion Dinamico/Estatico con HCD (ERASOR++)

**Paper**: ERASOR++ (Zhang & Zhang, 2024) — Height Coding Descriptor

**Que aporta**: Distinguir objetos detectados como **dinamicos** (coches en movimiento, peatones) o **estaticos** (edificios, muros, postes). Util para navegacion (los dinamicos requieren prediccion de trayectoria, los estaticos no).

**Como funciona HCD en ERASOR++**:
1. Divide el rango de altura en Nl=8 capas y codifica la ocupacion vertical de cada bin polar como un **bitmask de 8 bits**
2. Acumula un mapa estatico a lo largo del tiempo
3. Compara cada scan nuevo contra el mapa con **Height Stack Test (HST)**: bitwise AND entre capas del scan actual y del mapa
4. Si las capas superiores (sobre el suelo) no coinciden → objeto **dinamico** (algo cambio)
5. Si coinciden → objeto **estatico** (estructura permanente)

**Como integrarlo**:
1. Acumular puntos detectados como obstaculo en un mapa global (con egomotion compensation usando poses KITTI)
2. Para cada bin polar, mantener el bitmask de capas de altura del mapa
3. Comparar bitmask del scan actual vs mapa → clasificar clusters como dinamicos o estaticos
4. Opcionalmente, limpiar objetos dinamicos del mapa (proposito original de ERASOR++)

**Requisitos**: Mapa acumulado temporal + poses de odometria. **Sin mapa acumulado, HCD no tiene contra que comparar** — por eso en el pipeline actual (single-frame) no aporta mejora medible (+0.15% F1 maximo tras grid search de 28 combinaciones).

**Impacto estimado**: No mejora F1 binario (obs/no-obs) pero anade informacion semantica valiosa para navegacion autonoma. Permitiria tratar objetos dinamicos y estaticos de forma diferente en el planificador de trayectorias.

---

## Prioridad Sugerida

| Extension | Mejora F1? | Sin GPU? | Complejidad | Dependencias |
|-----------|-----------|----------|-------------|--------------|
| 1. Tracking (AB3DMOT) | Si (+1-3%) | Si | Media (~200 lineas) | Poses KITTI |
| 2. Prediccion posicion | Marginal | Si | Baja (incluido en Kalman) | Tracking (1) |
| 3. Oriented BBox + clasificacion | No (GT binario) | Si | Baja (~100 lineas) | Clusters Stage 3 |
| 4. HCD dinamico/estatico | No (GT binario) | Si | Media (~150 lineas) | Mapa acumulado + poses |

La extension con mayor impacto directo en metricas es el **tracking (1)**, que ademas habilita la prediccion (2) sin coste adicional. Las extensiones 3 y 4 no mejoran F1 binario pero enriquecen la salida del pipeline para aplicaciones downstream.
