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
  - Fase 1 (bin-wise): Si la normal del plano tiene `nz < 0.9`, el bin es sospechoso de ser pared
  - Fase 2 (point-wise): Voxel grid 2D (celdas 1.0m) con percentiles P95-P5 vectorizados. Rechaza puntos individuales con `delta_Z > 0.2m`. Optimizado de KDTree (2300ms) a voxel grid (74ms) con resultados equivalentes

**Metodo principal**: `stage1_complete(points)`

### Stage 2: Deteccion de Anomalias Delta-r

Clasifica cada punto comparando el rango medido con el rango esperado por el plano local:

```
delta_r = r_medido - r_esperado
r_esperado = -d / (n · direccion_rayo)

Si delta_r < -0.4m  →  OBSTACULO (algo bloquea el rayo antes del plano)
Si delta_r > +1.2m  →  VOID (hueco o depresion)
Si intermedio       →  GROUND normal
```

**Metodo principal**: `stage2_complete(points)` (ejecuta Stage 1 + delta-r)

### Stage 3: Filtrado por Clustering DBSCAN

Los obstaculos reales forman clusters densos (coche ~200 pts, persona ~50 pts). Los falsos positivos son puntos dispersos sin estructura espacial.

- Voxel downsampling (celdas 0.24m) antes de DBSCAN: reduce ~30k puntos a ~5k voxels centroides
- DBSCAN (eps=0.8m, min_samples=6) sobre voxels — parametros optimizados via grid search en SemanticKITTI (2880 combinaciones, train/val split)
- Clusters con >= 30 puntos originales → obstaculo real (se mantiene)
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
| **Grid search delta-r + DBSCAN** | `python3 tests/test_delta_r_dbscan_grid_search.py --mode full --workers 128 --stride 5` | Busca threshold_obs/void + eps/min_samples/min_pts optimos (2880 combos, train/val split) |
| **Grid search wall rejection** | `python3 tests/test_wall_rejection_grid_search.py --workers 128 --stride 5` | Busca slope/dz/radius optimos (100 combos, train/val split) |
| **Ablation completo** | `python3 tests/test_full_ablation.py --seq both --n_frames 10` | Contribucion de cada stage (acumulativo) |
| **Pipeline completo** | `python3 tests/test_full_pipeline_both_sequences.py` | Pipeline en ambas secuencias con metricas completas |
| **Ablation Stage 1** | `python3 tests/test_stage1_ablation.py --seq both --n_frames 10` | PW++ vanilla vs WR: ground seg + obs detection + timing |
| **Problema paredes** | `python3 tests/test_patchwork_wall_problem.py` | Cuantifica paredes mal clasificadas por Patchwork++ |

Metricas: Precision, Recall, F1, IoU, FP, FN + timing desglosado por stage.
Ground truth: SemanticKITTI labels (terrain excluido, moving objects incluidos).

---

## Resultados

### Evaluacion con protocolo SemanticKITTI

Parametros optimizados con grid search riguroso (protocolo SemanticKITTI: train 00-07,09-10 / val 08):
- **Delta-r + DBSCAN**: 2880 combinaciones (36 delta-r × 80 DBSCAN)
- **Wall Rejection**: 100 combinaciones (5 slope × 5 dz × 4 radius)
- Sin overfitting: mismos parametros optimos en train y val

### Resultados en Val (seq 08) — resultado reportable

| Configuracion | F1 | IoU | P | R |
|---------------|------|------|------|------|
| PW++ vanilla (sin wall rejection) | 91.94% | 85.07% | — | — |
| + Wall Rejection (slope=0.9, dz=0.2, r=0.3) | 92.20% | 85.53% | 90.34% | 94.14% |
| + delta-r optimizado (thr_obs=-0.4, thr_void=1.2) | 92.20% | 85.53% | 90.34% | 94.14% |
| **+ DBSCAN (eps=0.8, ms=12, mp=30) — pipeline final** | **93.25%** | **87.35%** | **91.18%** | **95.41%** |

### Parametros optimos (grid search)

| Parametro | Valor | Stage |
|-----------|-------|-------|
| wall_rejection_slope | 0.9 | Stage 1 |
| wall_height_diff_threshold | 0.2 m | Stage 1 |
| wall_kdtree_radius | 0.3 m | Stage 1 |
| threshold_obs | -0.4 m | Stage 2 |
| threshold_void | 1.2 m | Stage 2 |
| cluster_eps | 0.8 m | Stage 3 |
| cluster_min_samples | 12 | Stage 3 |
| cluster_min_pts | 30 | Stage 3 |

### Stages descartados tras ablation study

| Configuracion descartada | Media F1 | Motivo |
|--------------------------|----------|--------|
| Stage 2 → Bayes | 87.4% | Inercia temporal retrasa deteccion, genera fantasmas. Solo util con ruido transitorio (lluvia/polvo) que no existe en KITTI |
| Stage 2 → Bayes → DBSCAN | 88.2% | Bayes sigue empeorando respecto a Stage 2 → DBSCAN directo |
| Stage 2 → Bayes → Shadow → DBSCAN | 88.5% | Shadow apenas aporta (+0.3%), no justifica coste computacional |

Ver `lidar_pipeline_suite_with_bayes.py` para la version con filtro Bayesiano.

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

### Papers base (no directamente comparables)

Dewan et al. y OccAM usan datasets/metricas diferentes (KITTI Tracking, 3-class, range image). No se comparan numeros directamente. Este trabajo se inspira en ellos y aplica mejoras arquitectonicas:

| Aspecto | Dewan et al. | Este trabajo |
|---------|-------------|--------------|
| Representacion | Range image 64×870 (compresion 20:1) | Per-point 3D (sin compresion) |
| Ground seg. | No especificado | Patchwork++ + wall rejection hibrido |
| Post-filtrado | Ninguno | DBSCAN cluster filtering |
| Filtro temporal | Bayesian (componente principal) | Evaluado y descartado (ablation study) |
| Dataset | KITTI Tracking (3 clases) | SemanticKITTI (binario, protocolo estandar) |

### Comparacion directa (pendiente — mismo dataset, misma metrica)

Evaluacion en SemanticKITTI seq 08 (val), metrica binaria obstaculo/no-obstaculo:

| Metodo | Tipo | F1 (val) | IoU (val) | GPU | Entrenamiento |
|--------|------|----------|-----------|-----|---------------|
| RANSAC baseline | Geom. | pendiente | pendiente | No | No |
| Patchwork++ vanilla | Geom. | 91.94% | 85.07% | No | No |
| GroundGrid | Geom. | pendiente | pendiente | No | No |
| **Este trabajo** | **Geom.** | **93.25%** | **87.35%** | **No** | **No** |
| RangeNet++ | DL | pendiente | pendiente | Si | Si |
| Cylinder3D | DL | pendiente | pendiente | Si | Si |

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
│   ├── test_delta_r_dbscan_grid_search.py  # Grid search delta-r + DBSCAN (paralelo, train/val)
│   ├── test_wall_rejection_grid_search.py  # Grid search wall rejection (paralelo, train/val)
│   ├── test_full_pipeline_both_sequences.py
│   ├── test_patchwork_wall_problem.py
│   └── archive/                        # Tests obsoletos
│
├── data_odometry_velodyne/              # SemanticKITTI velodyne (seq 00-10)
├── data_odometry_labels/                # SemanticKITTI labels (seq 00-10)
├── data_paths.py                        # Modulo centralizado de rutas
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
2. **Dewan et al.** (IROS 2018) — Evaluado pero descartado: Bayesian temporal filter no mejora en buen tiempo (ver ablation study)
3. **OccAM** (Schinagl et al., CVPR 2022) — Evaluado pero descartado: Shadow validation aporta solo +0.3% F1

### Papers Evaluados sin Impacto

- **ERASOR++** (Zhang & Zhang, 2024) — HCD (Height Coding Descriptor) evaluado con grid search de 28 combinaciones per-point y 9 per-bin (ERASOR++ style). Maximo +0.15% F1. HCD esta disenado para comparar scans contra un mapa acumulado (dynamic object removal), no para deteccion single-frame. Sin mapa temporal, no tiene contra que comparar. Eliminado del pipeline.
