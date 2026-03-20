# LiDAR Obstacle Detection Pipeline

Sistema de deteccion de obstaculos 3D basado en LiDAR para navegación autónoma. Pipeline de 3 etapas que opera sin GPU y sin entrenamiento, usando geometria 3D pura. Evaluado con ground truth de SemanticKITTI.

**Tipo**: Trabajo de Fin de Grado (TFG)
**Lenguaje**: Python (comentarios en español)
**Ultima actualizacion**: 2026-03-20

---

## Pipeline: 3 Etapas

```
Point Cloud (128k puntos, Velodyne HDL-64E)
       |
  [Stage 1] Ground Segmentation (Patchwork++ + Wall Rejection)
       |
  [Stage 2] Delta-r Anomaly Detection
       |
  [Stage 3] DBSCAN Cluster Filtering
       |
  Output: obs_mask (N,), likelihood (N,), cluster_labels (N,)
```

### Stage 1: Segmentacion de Suelo + Rechazo de Paredes

Separa los puntos ground de los non-ground.

- **Patchwork++ (submodulo C++)**: Divide el espacio polar en 4 zonas concentricas (CZM). Para cada bin, ajusta un plano local con RANSAC. Un punto es ground si su distancia al plano es menor que el umbral (`th_dist = 0.2m`).

- **Rechazo hibrido de paredes**: Patchwork++ clasifica mal las bases de paredes y objetos verticales como suelo. Se corrige en dos fases:
  - Fase 1 (bin-wise): Si la normal del plano tiene `nz < 0.7`, el bin es sospechoso de ser pared
  - Fase 2 (point-wise): Voxel grid 2D (celdas 1.0m) con percentiles P95-P5 vectorizados. Rechaza puntos individuales con `delta_Z > 0.3m`. Optimizado de KDTree (2300ms) a voxel grid (74ms) con resultados equivalentes

**Metodo principal**: `stage1_complete(points)`

### Stage 2: Deteccion de Anomalias Delta-r

Clasifica cada punto comparando el rango medido con el rango esperado por el plano local:

```
delta_r = r_medido - r_esperado
r_esperado = -d / (n · direccion_rayo)

Si delta_r < -0.5m  →  OBSTACULO (algo bloquea el rayo antes del plano)
Si delta_r > +0.8m  →  VOID (hueco o depresion)
Si intermedio       →  GROUND normal
```

**Metodo principal**: `stage2_complete(points)` (ejecuta Stage 1 + delta-r)

### Stage 3: Filtrado por Clustering DBSCAN

Los obstaculos reales forman clusters densos (coche ~200 pts, persona ~50 pts). Los falsos positivos son puntos dispersos sin estructura espacial.

- DBSCAN (eps=0.8m, min_samples=8) — parametros optimizados via grid search (108 combinaciones)
- Clusters con >= 30 puntos → obstaculo real (se mantiene)
- Clusters pequenos o ruido → FP probable (se elimina)

**Metodo principal**: `stage3_cluster_filtering(points, stage2_result)` o `stage3_complete(points)` (pipeline completo)

---

## Como Ejecutar

### Requisitos

```bash
# Compilar Patchwork++ (desde raiz del workspace)
cd ~/lidar_ws/TFG-LiDAR-Geometry
colcon build --packages-select patchworkpp
source install/setup.bash
```

### Visualizacion en RViz (run_pipeline_viz.py)

Visualiza cada stage con colores (verde=ground, rojo=obstaculo, azul=void, amarillo=paredes, gris=incierto):

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea

# Visualizar todos los stages en scan 10 de seq 04
python3 run_pipeline_viz.py --seq 04 --scan 10

# Procesar frames 0-10 (acumula temporal) y visualizar stages 1 y 2
python3 run_pipeline_viz.py --seq 00 --scan_start 0 --scan_end 10 --stages 1 2

# Procesar frames 0-10 (acumula temporal) y visualizar todos los stages
python3 run_pipeline_viz.py --seq 00 --scan_start 0 --scan_end 10

# Sin lanzar RViz automaticamente
python3 run_pipeline_viz.py --seq 04 --scan 0 --no-rviz
```

Topics publicados: `/stage1_cloud`, `/stage2_cloud`, `/stage3_cloud`, `/gt_cloud`


### Uso como Libreria

```python
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

config = PipelineConfig(verbose=True)
pipeline = LidarPipelineSuite(config)

# Single frame (Stage 2 solo)
result = pipeline.stage2_complete(points)

# Pipeline completo (Stage 1 + 2 + 3 DBSCAN)
result = pipeline.stage3_complete(points)

obs_mask = result['obs_mask']          # (N,) bool
likelihood = result['likelihood']      # (N,) log-odds
clusters = result['cluster_labels']    # (N,) int
```

---

## Tests

Todos los tests estan en `tests/` y se ejecutan desde `sota_idea/`:

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
```

| Test | Comando | Que mide |
|------|---------|----------|
| **Ablation completo** | `python3 tests/test_full_ablation.py --seq both --n_frames 10` | Contribucion de cada stage (acumulativo) |
| **Pipeline completo** | `python3 tests/test_full_pipeline_both_sequences.py` | Pipeline en ambas secuencias con metricas completas |
| **Ablation Stage 1** | `python3 tests/test_stage1_ablation.py --seq both --n_frames 10` | PW++ vanilla vs WR: ground seg + obs detection + timing |
| **Grid search DBSCAN** | `python3 tests/test_dbscan_grid_search.py --seq both --n_frames 10` | Busca eps/min_samples/min_pts optimos para DBSCAN |
| **Problema paredes** | `python3 tests/test_patchwork_wall_problem.py` | Cuantifica paredes mal clasificadas por Patchwork++ |

Metricas: Precision, Recall, F1, IoU, FP, FN + timing desglosado por stage.
Ground truth: SemanticKITTI labels (terrain excluido, moving objects incluidos).

---

## Resultados

Evaluado en KITTI Seq 00 (urbano, ~27 km/h) y Seq 04 (autopista, ~47 km/h), 10 frames.

### Ablation Acumulativo

| Configuracion | Seq 04 F1 | Seq 00 F1 | Media F1 | Tiempo total |
|---------------|-----------|-----------|----------|--------------|
| Stage 2 (baseline single-frame) | 86.7% | 92.4% | 89.5% | ~87 ms |
| **Stage 2 → 3 DBSCAN (pipeline final)** | **88.2%** | **93.0%** | **90.6%** | **~377 ms** |

Nota: Se evaluaron tambien Stage 3 (Bayesian temporal, Dewan et al.) y Stage 4 (Shadow validation, OccAM) pero fueron descartados tras ablation study:

| Configuracion descartada | Media F1 | Motivo |
|--------------------------|----------|--------|
| Stage 2 → Bayes | 87.4% | Inercia temporal retrasa deteccion, genera fantasmas. Solo util con ruido transitorio (lluvia/polvo) que no existe en KITTI |
| Stage 2 → Bayes → DBSCAN | 88.2% | Bayes sigue empeorando respecto a Stage 2 → DBSCAN directo |
| Stage 2 → Bayes → Shadow → DBSCAN | 88.5% | Shadow apenas aporta (+0.3%), no justifica coste computacional |

Ver `lidar_pipeline_suite_with_bayes.py` para la version con filtro Bayesiano.

### Impacto de Wall Rejection (Stage 1 Ablation)

Comparacion de Patchwork++ vanilla vs Stage 1 con rechazo de paredes, 10 frames por secuencia.

| Configuracion | Mean Gnd F1 | Mean Obs F1 | Mean Leak % | Mean Stage 1 ms |
|---------------|-------------|-------------|-------------|-----------------|
| Patchwork++ vanilla | 95.3% | 89.1% | 12.5% | 26 ms |
| **PW++ + Wall Rejection** | **96.0%** | **89.7%** | **9.1%** | **74 ms** |

Wall Rejection reduce el obstacle leak (obstaculos GT clasificados como ground) de 12.5% a 9.1% y mejora Obs F1 en +0.6%. Mayor impacto en Seq 00 (urbano): leak baja de 14.0% a 10.8%.

Optimizacion: La Fase 2 (point-wise) se reemplazo de KDTree `query_ball_point()` (2300 ms) por voxel grid 2D con percentiles vectorizados (74 ms), logrando 32x speedup con -0.2% F1 de diferencia.

Nota: HCD (Height Coding Descriptor, ERASOR++) fue evaluado pero su impacto en la clasificacion binaria (obs_mask) es nulo sin filtro Bayesiano. HCD solo modula magnitudes de likelihood, que no cambian la clasificacion final por umbral. Se mantiene en el codigo como opcion pero no aporta mejora medible al pipeline actual.

### Pipeline Completo (Stage 2 → DBSCAN)

| Secuencia | Precision | Recall | F1 | IoU |
|-----------|-----------|--------|------|------|
| Seq 04 (highway) | 83.9% | 93.0% | 88.2% | 78.9% |
| Seq 00 (urbano) | 96.2% | 89.9% | 93.0% | 86.9% |
| **Media** | **90.1%** | **91.5%** | **90.6%** | **82.9%** |

Timing por stage (media): Stage 1+2 = ~80 ms, Stage 3 (DBSCAN) = ~300 ms, **Total = ~377 ms**

---

## Problemas Detectados y Superados

### 1. Paredes Clasificadas como Suelo (Patchwork++)

**Problema**: Patchwork++ ajusta planos por bins CZM. Si un bin contiene la base de una pared, el plano ajustado es vertical y la base se clasifica como ground. Afecta edificios, muros, y la parte baja de vehiculos.

**Solucion**: Rechazo hibrido en dos fases. Fase 1 (bin-wise) detecta bins sospechosos por normal vertical insuficiente. Fase 2 (point-wise) refina con voxel grid 2D (percentiles P95-P5) para no rechazar bins completos. Rescata ~847 puntos/frame. Optimizado de KDTree (2300ms) a voxel grid vectorizado (74ms).

### 2. Filtro Temporal Bayesiano No Mejora en KITTI

**Problema**: Se implemento el filtro temporal Bayesiano de Dewan et al. (per-point con KDTree, gamma adaptativo, egomotion compensation). Tras grid search de 360 combinaciones de parametros, 0 mejoraron sobre el baseline Stage 2. F1 bajaba de 90.8% a 88.2%.

**Causa**: El filtro fue disenado para filtrar ruido transitorio (polvo, lluvia, humo) que no existe en KITTI (buen tiempo). La inercia temporal introduce retraso en deteccion y persistencia fantasma sin beneficio compensatorio.

**Decision**: Eliminar Bayes del pipeline principal. Pipeline optimo: Stage 2 → DBSCAN directo (F1=90.8%). Version con Bayes preservada en `lidar_pipeline_suite_with_bayes.py`.


## Comparacion con Papers Base

| Metodo | Tipo | F1 | GPU | Entrenamiento | Limitacion principal |
|--------|------|------|-----|---------------|----------------------|
| Cylinder3D | CNN | ~97% | Si | Si | Segmentacion semantica completa (19 clases), problema diferente |
| RangeNet++ | CNN | ~84% | Si | Si | Solo range image, pierde resolucion 3D |
| Dewan et al. | Bayesian | ~83% | No | No | Range image (compresion 20:1), sin gamma, sin egomotion |
| OccAM | Shadow | ~85% | No | Parcial | Solo validacion por sombra, sin temporal |
| **Este trabajo** | **Geometria** | **90.6%** | **No** | **No** | Single-frame, ~377ms (sin filtro temporal) |

**Mejoras respecto a Dewan et al.**: Per-point 3D (sin range image), wall rejection hibrido con voxel grid, DBSCAN cluster filtering optimizado (grid search 108 combinaciones). El filtro temporal Bayesiano y HCD fueron evaluados pero descartados (ver ablation study). Resultado: +7.6% F1.

---

## Estructura del Proyecto

```
sota_idea/
├── lidar_pipeline_suite.py             # Pipeline principal (Stages 1-3, libreria)
├── lidar_pipeline_suite_with_bayes.py  # Version con filtro Bayesiano (backup)
├── run_pipeline_viz.py                 # Visualizacion por stages en RViz
├── range_projection.py                 # Pipeline como nodo ROS 2
├── stage1_visualizer.py                # Visualizacion Stage 1 aislado
│
├── tests/
│   ├── test_full_ablation.py           # Ablation acumulativo por stages
│   ├── test_stage1_ablation.py         # PW++ vanilla vs Wall Rejection + timing
│   ├── test_dbscan_grid_search.py      # Grid search parametros DBSCAN
│   ├── test_full_pipeline_both_sequences.py
│   ├── test_patchwork_wall_problem.py
│   └── archive/                        # Tests obsoletos
│
├── data_kitti/
│   ├── 00/ y 00_labels/                # KITTI seq 00 (urbano)
│   └── 04/ y 04_labels/                # KITTI seq 04 (highway)
│
├── Paso_1/                             # Exploracion inicial de Patchwork++
│   ├── run_patchwork_viz.sh
│   └── visualize_patchwork_rviz.py
│
└── papers/                             # PDFs de papers base
```

---

## Papers Implementados

1. **Patchwork++** (Lee et al., RA-L/IROS 2022) — Stage 1: Ground segmentation con CZM
2. **ERASOR++** (Lim et al., ICRA 2023) — Evaluado: HCD no aporta mejora medible sin filtro Bayesiano (ver ablation study)
3. **Dewan et al.** (IROS 2018) — Evaluado pero descartado: Bayesian temporal filter no mejora en buen tiempo (ver ablation study)
4. **OccAM** (Schinagl et al., CVPR 2022) — Evaluado pero descartado: Shadow validation aporta solo +0.3% F1
