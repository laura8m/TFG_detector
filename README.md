# LiDAR Obstacle Detection Pipeline

Sistema de deteccion de obstaculos 3D basado en LiDAR para navegacion autonoma. Combina Patchwork++ con rechazo hibrido de paredes y deteccion de anomalias delta-r conservador. Opera sin GPU y sin entrenamiento, usando geometria 3D pura. Evaluado con ground truth de SemanticKITTI (protocolo estandar: train 00-07,09-10 / val 08).

**Tipo**: Trabajo de Fin de Grado (TFG)
**Lenguaje**: Python (comentarios en espanol)
**Ultima actualizacion**: 2026-03-22

---

## Pipeline

```
Point Cloud (128k puntos, Velodyne HDL-64E)
       |
  [Stage 1] Ground Segmentation (Patchwork++ + Wall Rejection hibrido)
       |                         F1=95.31%  ~41 ms/frame
       |
  [Stage 2] Delta-r conservador (opcional, desactivado por defecto)
       |                         F1=95.28%  ~47 ms/frame
       |
  non-ground = obstaculo
```

**Pipeline optimo: Stage 1 solo (PW++ + WR).** Delta-r conservador esta disponible como opcion para conduccion real (detecta bordillos y hoyos) pero no mejora en SemanticKITTI (-0.03% F1).

### Stage 1: Segmentacion de Suelo + Rechazo de Paredes

Separa los puntos ground de los non-ground.

- **Patchwork++ (submodulo C++)**: Divide el espacio polar en 4 zonas concentricas (CZM). Para cada bin, ajusta un plano local con RANSAC. Un punto es ground si su distancia al plano es menor que el umbral.

- **Rechazo hibrido de paredes (contribucion original)**: Patchwork++ clasifica mal las bases de paredes y objetos verticales como suelo. Se corrige en dos fases:
  - Fase 1 (bin-wise): Si la normal del plano tiene `nz < 0.9`, el bin es sospechoso de ser pared
  - Fase 2 (point-wise): Voxel grid 2D (celdas 1.0m) con percentiles P95-P5 vectorizados. Rechaza puntos individuales con `delta_Z > 0.2m`
  - **Impacto**: +2.06% F1 sobre Patchwork++ vanilla (93.25% → 95.31%)

### Stage 2: Delta-r Conservador (opcional)

Rescata puntos ground que son obstaculos reales, solo en bins con plano fiable (nz >= 0.95). Nunca degrada detecciones de Stage 1 (solo rescate ground→obstaculo, nunca non-ground→ground).

- `delta_r < -0.8` → obstaculo positivo (algo sobresale del suelo)
- `delta_r > 1.5` → void/depresion (hoyo, bache)
- Solo en bins fiables (`min_nz >= 0.95`)

**En SemanticKITTI**: -0.03% F1 (detecta bordillos que el GT considera ground → FP).
**En conduccion real**: util para detectar bordillos infranqueables y depresiones del terreno. Activable con `enable_delta_r=True`.

### Stages evaluados y descartados

| Stage | F1 (val) | Delta vs WR solo | Motivo |
|-------|----------|-------------------|--------|
| DBSCAN cluster filtering | 95.27% | -0.04% | Elimina obstaculos pequenos reales (peatones lejanos, postes). No mejora, anade ~52ms |
| Filtro temporal Bayesiano | ~88% | -7% | Disenado para ruido transitorio (lluvia/polvo) que no existe en KITTI |
| Shadow validation (OccAM) | ~88.5% | — | Apenas aporta (+0.3%), no justifica coste computacional |

---

## Resultados (SemanticKITTI val, seq 08)

### Ablation Study (815 frames, stride=5)

| Configuracion | F1 | IoU | P | R | ms/frame |
|---------------|------|------|------|------|----------|
| PW++ vanilla | 93.25% | 87.35% | 98.66% | 88.39% | 32.3 |
| **PW++ + Wall Rejection** | **95.31%** | **91.03%** | **94.83%** | **95.79%** | **40.9** |
| PW++ + WR + delta-r conservador | 95.28% | 90.98% | 93.77% | 96.84% | 46.6 |
| PW++ + WR + delta-r + DBSCAN | 95.27% | 90.96% | 94.06% | 96.51% | 98.4 |

**Wall Rejection: +2.06% F1, +3.68% IoU, +7.40% Recall sobre PW++ vanilla.**

### Analisis por clase (Recall, seq 00, 454 frames)

| Clase | N puntos | PW++ vanilla | PW++ + WR | Delta |
|-------|----------|-------------|-----------|-------|
| building | 10,795,752 | 72.85% | 89.82% | **+16.97%** |
| fence | 2,161,746 | 78.75% | 91.02% | **+12.27%** |
| vegetation | 6,660,555 | 90.54% | 94.38% | +3.84% |
| trunk | 1,018,547 | 89.91% | 93.02% | +3.11% |
| pole | 326,483 | 85.87% | 89.45% | +3.58% |
| person | 119,283 | 85.86% | 89.22% | +3.36% |
| car | 4,998,870 | 94.98% | 96.41% | +1.43% |

WR mejora mas en **buildings (+17%) y fences (+12%)**: las estructuras verticales que PW++ confunde con suelo.

### Analisis por distancia (F1, seq 00, 454 frames)

| Rango | PW++ vanilla | PW++ + WR | Delta |
|-------|-------------|-----------|-------|
| 0-10m | 96.09% | 97.03% | +0.94% |
| 10-20m | 93.43% | 96.30% | +2.87% |
| 20-30m | 89.44% | 94.69% | +5.25% |
| 30-40m | 83.34% | 91.24% | +7.90% |
| 40-60m | 74.17% | 84.31% | **+10.14%** |
| 60-80m | 59.21% | 66.88% | +7.67% |

WR mejora mas **a distancia** (40-60m: +10.14%). A distancia, PW++ ajusta peores planos (menos puntos) y clasifica mas paredes como suelo. WR corrige eso.

### Stage 1 detallado (PW++ vanilla vs PW++ + WR)

| Metrica | PW++ vanilla | PW++ + WR | Delta |
|---------|-------------|-----------|-------|
| Ground F1 | 94.1% | 96.1% | +2.0% |
| Obstacle Leak | 13.2% | 5.8% | **-7.4%** |
| Obstacle F1 | 92.4% | 94.5% | +2.1% |
| Stage 1 ms | 37.4ms | 52.2ms | +14.8ms |

WR reduce el obstacle leak a menos de la mitad: 13.2% → 5.8%.

### Parametros optimos (grid search)

Optimizados con protocolo SemanticKITTI (tuning en train, evaluacion en val):

| Parametro | Valor | Stage | Grid search |
|-----------|-------|-------|-------------|
| wall_rejection_slope | 0.9 | Stage 1 | 100 combos (5×5×4) |
| wall_height_diff_threshold | 0.2 m | Stage 1 | |
| wall_kdtree_radius | 0.3 m | Stage 1 | |
| threshold_obs | -0.8 | Stage 2 | 144 combos (6×6×4) |
| threshold_void | 1.5 m | Stage 2 | |
| delta_r_min_nz | 0.95 | Stage 2 | |

---

## Como Ejecutar

### Requisitos

```bash
# Compilar Patchwork++ (desde raiz del workspace)
cd ~/lidar_ws/TFG-LiDAR-Geometry
colcon build --packages-select patchworkpp
source install/setup.bash
```

### Visualizacion en RViz

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea

# Frame unico
python3 run_pipeline_viz.py --seq 00 --scan 50

# Rango de frames
python3 run_pipeline_viz.py --seq 04 --scan_start 0 --scan_end 10

# Solo stages 1 y 2
python3 run_pipeline_viz.py --seq 00 --scan 50 --stages 1 2

# Sin lanzar RViz automaticamente
python3 run_pipeline_viz.py --seq 04 --scan 0 --no-rviz
```

Colores: verde=ground, rojo=obstaculo, azul=void (hoyo), amarillo=paredes rechazadas.
Topics: `/stage1_cloud`, `/stage2_cloud`, `/stage3_cloud`, `/gt_cloud`

### Uso como Libreria

```python
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

# Pipeline optimo (default): PW++ + WR, sin delta-r
config = PipelineConfig()
pipeline = LidarPipelineSuite(config)
result = pipeline.stage2_complete(points)
obs_mask = result['obs_mask']  # (N,) bool

# Con delta-r conservador (para conduccion real)
config = PipelineConfig(enable_delta_r=True)
pipeline = LidarPipelineSuite(config)
result = pipeline.stage2_complete(points)
obs_mask = result['obs_mask']
```

---

## Tests

Todos los tests estan en `tests/` y se ejecutan desde `sota_idea/`:

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
```

### Tests principales (resultados para el TFG)

| Test | Comando | Que mide |
|------|---------|----------|
| **Ablation SemanticKITTI** | `python3 tests/test_stage_ablation_semantickitti.py --stride 5` | Ablation incremental en val (seq 08): PW++ → +WR → +delta-r → +DBSCAN |
| **Ablation Stage 1** | `python3 tests/test_stage1_ablation.py --seq both --n_frames 30` | PW++ vanilla vs WR: ground seg + obstacle leak + timing |
| **Analisis por clase/distancia** | `python3 tests/test_per_class_distance.py --seq 08 --stride 5` | Recall por tipo de obstaculo + F1 por rango de distancia |

### Grid searches (optimizacion de parametros, ejecutar en Mazinger)

| Test | Comando | Combos |
|------|---------|--------|
| **Grid search WR** | `python3 tests/test_wall_rejection_grid_search.py --workers 128 --stride 5 --no_dbscan` | 100 (5×5×4) |
| **Grid search delta-r** | `python3 tests/test_delta_r_dbscan_grid_search.py --mode delta_r --workers 128 --stride 5 --conservative` | 144 (6×6×4) |
| **Grid search DBSCAN** | `python3 tests/test_delta_r_dbscan_grid_search.py --mode dbscan --workers 128 --stride 5 --conservative` | 80 |

**Orden correcto de grid searches** (cadena de dependencias):
1. WR sin delta-r → params WR optimos
2. Delta-r con WR fijado → params delta-r optimos
3. Ablation final con todos los params

### Tests auxiliares

| Test | Comando | Que mide |
|------|---------|----------|
| Delta-r conservador vs original | `python3 tests/test_delta_r_conservative.py --seq both` | Compara 3 modos: Stage 1 solo, +delta-r original, +delta-r conservador |
| Problema paredes PW++ | `python3 tests/test_patchwork_wall_problem.py` | Cuantifica paredes mal clasificadas por Patchwork++ |

---

## Problemas Detectados y Superados

### 1. Paredes Clasificadas como Suelo (Patchwork++)

**Problema**: Patchwork++ ajusta planos por bins CZM. Si un bin contiene la base de una pared, el plano ajustado es vertical y la base se clasifica como ground. Afecta especialmente buildings (+17% recall con WR) y fences (+12%).

**Solucion**: Rechazo hibrido en dos fases. Fase 1 (bin-wise) detecta bins sospechosos por normal vertical insuficiente (nz < 0.9). Fase 2 (point-wise) refina con voxel grid 2D (percentiles P95-P5) para no rechazar bins completos. Reduce obstacle leak de 13.2% a 5.8%.

### 2. Delta-r Original Empeora el Pipeline

**Problema**: Delta-r compara rango medido con rango esperado por plano RANSAC. Los planos ruidosos generan delta-r incorrecto → FPs. Ademas, reclasifica non-ground correctos como ground.

**Solucion**: Modo conservador — solo rescata ground→obstaculo en bins fiables (nz >= 0.95). Nunca degrada Stage 1. Mejora +0.32% sobre delta-r original, pero no supera WR solo (-0.03%). Desactivado por defecto, disponible para conduccion real donde bordillos y hoyos son relevantes.

### 3. Evaluacion con Labels Incorrectas (corregido)

**Problema**: Labels 52 (other-structure) y 99 (other-object) estaban incluidas como obstaculos. En SemanticKITTI learning_map, ambas mapean a clase 0 (unlabeled/ignored). Labels 0 y 1 se contaban como "non-obstacle" generando FPs.

**Solucion**: Eliminadas de OBSTACLE_LABELS, anadidas a IGNORE_LABELS = {0, 1, 52, 99}. Todos los tests usan valid_mask que excluye puntos ignorados de las metricas. ~3.15% de puntos afectados.

---

## Estructura del Proyecto

```
sota_idea/
├── lidar_pipeline_suite.py             # Pipeline principal (Stages 1-2, libreria)
├── run_pipeline_viz.py                 # Visualizacion por stages en RViz
├── pipeline_viz.rviz                   # Configuracion RViz (3 stages)
├── data_paths.py                       # Modulo centralizado de rutas
│
├── tests/
│   ├── test_stage_ablation_semantickitti.py  # Ablation incremental (val seq 08)
│   ├── test_stage1_ablation.py              # PW++ vanilla vs WR detallado
│   ├── test_per_class_distance.py           # Analisis por clase y distancia
│   ├── test_delta_r_dbscan_grid_search.py   # Grid search delta-r + DBSCAN
│   ├── test_wall_rejection_grid_search.py   # Grid search wall rejection
│   ├── test_delta_r_conservative.py         # Comparativa conservador vs original
│   └── test_patchwork_wall_problem.py       # Diagnostico paredes PW++
│
├── test_data/sequences/                # SemanticKITTI (velodyne + labels, seq 00-10)
│
└── papers/                             # PDFs de papers base
```

---

## Papers Implementados

1. **Patchwork++** (Lee et al., RA-L/IROS 2022) — Stage 1: Ground segmentation con CZM
2. **Dewan et al.** (IROS 2018) — Evaluado pero descartado: Bayesian temporal filter no mejora en buen tiempo
3. **OccAM** (Schinagl et al., CVPR 2022) — Evaluado pero descartado: Shadow validation aporta solo +0.3% F1
