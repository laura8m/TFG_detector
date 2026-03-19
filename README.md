# LiDAR Obstacle Detection Pipeline

Sistema de deteccion de obstaculos 3D basado en LiDAR para navegación autónoma. Pipeline de 5 etapas que opera sin GPU y sin entrenamiento, usando geometria 3D y estadistica Bayesiana. Evaluado con ground truth de SemanticKITTI.

**Tipo**: Trabajo de Fin de Grado (TFG)
**Lenguaje**: Python (comentarios en español)
**Ultima actualizacion**: 2026-03-18

---

## Pipeline: 5 Etapas

```
Point Cloud (128k puntos, Velodyne HDL-64E)
       |
  [Stage 1] Ground Segmentation
       |
  [Stage 2] Delta-r Anomaly Detection
       |
  [Stage 3] Bayesian Temporal Filter
       |
  [Stage 4] Shadow Validation
       |
  [Stage 5] DBSCAN Cluster Filtering
       |
  Output: obs_mask (N,), belief (N,), cluster_labels (N,)
```

### Stage 1: Segmentacion de Suelo + Rechazo de Paredes + HCD

Separa los puntos ground de los non-ground.

- **Patchwork++ (submodulo C++)**: Divide el espacio polar en 4 zonas concentricas (CZM). Para cada bin, ajusta un plano local con RANSAC. Un punto es ground si su distancia al plano es menor que el umbral (`th_dist = 0.2m`).

- **Rechazo hibrido de paredes**: Patchwork++ clasifica mal las bases de paredes y objetos verticales como suelo. Se corrige en dos fases:
  - Fase 1 (bin-wise): Si la normal del plano tiene `nz < 0.7`, el bin es sospechoso de ser pared
  - Fase 2 (point-wise): KDTree batch `query_ball_point()` (radio 0.5m), rechaza solo puntos individuales con `delta_Z > 0.3m`

- **HCD (ERASOR++)**: Height Coding Descriptor. Mide la altura relativa de cada punto ground respecto a su plano local. Normalizado con `tanh(z_rel / 0.3)`. Permite distinguir rampas suaves de escalones verticales.

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

**Fusion HCD** (opcional): Modula la likelihood segun la geometria vertical del HCD. Obstaculos con HCD alto reciben mayor confianza (+4.0 log-odds), ground plano recibe mayor supresion (-2.5).

**Metodo principal**: `stage2_complete(points)` (ejecuta Stage 1 + delta-r)

### Stage 3: Filtro Temporal Bayesiano Per-Point

Acumula evidencia temporal entre frames. Un obstaculo real persiste frame tras frame; el ruido es transitorio.

- **Asociacion por KDTree**: Cada punto del frame actual busca su punto mas cercano del frame anterior (compensando egomotion con poses KITTI). Si la distancia es < 2.0m, hereda el belief previo.

- **Gamma adaptativo por velocidad**: A alta velocidad, el warping KDTree introduce errores de alineacion → los falsos positivos se acumulan. Solucion: gamma baja automaticamente cuando la velocidad sube.
  ```
  Velocidad < 0.8 m/frame (~30 km/h):  gamma = 0.7 (acumulacion normal)
  Velocidad > 2.0 m/frame (~72 km/h):  gamma = 0.0 (single-frame puro)
  Intermedio: interpolacion lineal
  ```

- **Protecciones**: Descarte de warping si < 30% de puntos se asocian. Depth jump check que resetea belief si el rango cambio drasticamente (objeto se fue).

- **Update Bayesiano** (Dewan Eq. 9 con gamma):
  ```
  belief_t = likelihood_t + gamma * (belief_{t-1} - l0) + l0
  ```

**Metodo principal**: `stage3_per_point(points, delta_pose)`

### Stage 4: Validacion por Sombra Geometrica

Un obstaculo solido (coche, muro) bloquea TODOS los rayos laser, creando una zona de void detras. Particulas dispersas (polvo, lluvia) dejan pasar algunos rayos, permitiendo ver ground detras.

- KDTree angular sobre ground points
- Para cada candidato obstaculo, busca ground en cono angular (0.035 rad) dentro de la sombra geometrica: `L_shadow = r_obs * h_obs / (h_sensor - h_obs)`
- Ground encontrado (>= 3 pts): oclusion parcial → suppress -1.0 log-odds
- Sin ground, con vecinos: oclusion completa → boost +1.0 log-odds
- Decaimiento exponencial por distancia (menos confianza a mayor rango)

**Metodo principal**: `stage4_shadow_validation(points, stage3_result)`

### Stage 5: Filtrado por Clustering DBSCAN

Los obstaculos reales forman clusters densos (coche ~200 pts, persona ~50 pts). Los falsos positivos son puntos dispersos sin estructura espacial.

- DBSCAN (eps=0.5m, min_samples=5)
- Clusters con >= 15 puntos → obstaculo real (se mantiene)
- Clusters pequenos o ruido → FP probable (se elimina)

**Metodo principal**: `stage5_cluster_filtering(points, stage4_result)`

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

Topics publicados: `/stage1_cloud`, `/stage2_cloud`, `/stage3_cloud`, `/stage4_cloud`, `/stage5_cloud`, `/gt_cloud`


### Uso como Libreria

```python
from lidar_pipeline_suite import LidarPipelineSuite, PipelineConfig

config = PipelineConfig(verbose=True)
pipeline = LidarPipelineSuite(config)

# Single frame (Stage 2 solo)
result = pipeline.stage2_complete(points)

# Multi-frame (pipeline completo con temporal)
for i, (pts, delta_pose) in enumerate(frames):
    result = pipeline.stage5_per_point(pts, delta_pose=delta_pose)

obs_mask = result['obs_mask']     # (N,) bool
belief = result['belief']         # (N,) log-odds
clusters = result['cluster_labels']  # (N,) int
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
| **Ablation HCD** | `python3 tests/test_full_ablation.py --hcd --seq both --n_frames 10` | Impacto de HCD (con vs sin) |
| **Gamma adaptativo** | `python3 tests/test_adaptive_gamma.py --seq both --n_frames 10` | Gamma fijo vs adaptativo a distintas velocidades |
| **Shadow validation** | `python3 tests/test_stage4_shadow_validation.py --seq 04 --n_frames 10` | Efectividad del filtro de sombra |
| **DBSCAN filtering** | `python3 tests/test_stage5_cluster_filtering.py --seq 04 --n_frames 10` | Ablation de min_pts (5, 10, 15, 25, 50) |
| **Pipeline completo** | `python3 tests/test_full_pipeline_both_sequences.py` | Pipeline en ambas secuencias con metricas completas |
| **Problema paredes** | `python3 tests/test_patchwork_wall_problem.py` | Cuantifica paredes mal clasificadas por Patchwork++ |

Metricas: Precision, Recall, F1, IoU, FP, FN + timing desglosado por stage.
Ground truth: SemanticKITTI labels (terrain excluido, moving objects incluidos).

---

## Resultados

Evaluado en KITTI Seq 00 (urbano, ~27 km/h) y Seq 04 (autopista, ~47 km/h), 10 frames.

### Ablation Acumulativo

| Configuracion | Seq 04 F1 | Seq 00 F1 | Media F1 |
|---------------|-----------|-----------|----------|
| Stage 2 (baseline single-frame) | 87.3% | 92.5% | 89.9% |
| Stage 2 → 3 (+ Bayesian temporal) | 85.9% | 88.8% | 87.4% |
| Stage 2 → 3 → 4 (+ Shadow) | 86.6% | 88.8% | 87.7% |
| **Stage 2 → 3 → 4 → 5 (completo)** | **88.0%** | **89.1%** | **88.5%** |
| Stage 2 → 3 → 5 (sin Shadow) | 87.2% | 89.2% | 88.2% |

### Impacto de HCD (Height Coding Descriptor)

| Configuracion | Seq 04 F1 | Seq 00 F1 | Media F1 |
|---------------|-----------|-----------|----------|
| Stage 2 sin HCD | 86.7% | 87.9% | 87.3% |
| Stage 2 con HCD | 87.3% | 92.5% | 89.9% |
| **Diferencia** | **+0.57%** | **+4.65%** | **+2.6%** |
| Pipeline completo sin HCD | 87.7% | 88.7% | 88.2% |
| Pipeline completo con HCD | 88.0% | 89.1% | 88.5% |
| **Diferencia** | **+0.32%** | **+0.32%** | **+0.32%** |

HCD tiene mayor impacto en Seq 00 (urbano): reduce 11,861 FP en Stage 2 gracias a la mejor discriminacion de geometria vertical en entornos con edificios y paredes.

### Pipeline Completo

| Secuencia | Precision | Recall | F1 | FP | FN |
|-----------|-----------|--------|------|------|------|
| Seq 04 (highway) | 87.6% | 88.4% | 88.0% | 3,895 | 3,635 |
| Seq 00 (urbano) | 83.7% | 95.3% | 89.1% | 11,708 | 2,989 |

### Timing (ultimo frame, Seq 04)

| Stage | Tiempo | Descripcion |
|-------|--------|-------------|
| S1+S2 | ~2,100 ms | Patchwork++ + Wall Rejection + Delta-r |
| S3 | ~3,175 ms | KDTree warp + Bayesian update |
| S4 | ~188 ms | Shadow Validation |
| S5 | ~209 ms | DBSCAN Clustering |
| **Total** | **~5,670 ms** | Per frame |

---

## Problemas Detectados y Superados

### 1. Compresion 20:1 en Range Image

**Problema**: La implementacion inicial (Dewan et al.) operaba sobre range image 2D. 128k puntos se comprimen a ~6k pixels, perdiendo el 49.7% de obstaculos GT por colisiones de proyeccion.

**Solucion**: Operar per-point con KDTree en 3D. Cada punto mantiene su identidad sin comprimir. Stage 3 usa `cKDTree.query()` para asociacion temporal directa entre puntos 3D.

### 2. Temporal Filter sin Egomotion

**Problema**: Sin compensacion de egomotion, la asociacion KDTree entre frames consecutivos falla a alta velocidad. En seq 04 (1.3 m/frame), solo el 8% de puntos se asociaban correctamente.

**Solucion**: Transformar puntos del frame anterior al sistema de referencia actual usando poses KITTI (`delta_pose = inv(T_t) @ T_{t-1}`). Asociacion sube al 98.7%.

### 3. Belief sin Decaimiento (gamma)

**Problema**: La implementacion original usaba `belief = likelihood + belief_warped` (suma pura, Dewan Eq. 9 sin gamma). El belief crecia sin limite → FP explotaban de 7k a 24k en 10 frames.

**Solucion**: Introducir gamma como factor de inercia: `belief = likelihood + gamma * (belief_warped - l0) + l0`. Gamma controla cuanto pesa el pasado vs la observacion actual.

### 4. Gamma Fijo Destruye Highway

**Problema**: Con gamma=0.7 fijo, seq 04 (highway a 47 km/h) tenia F1=79.7%. A alta velocidad, los errores de alineacion del warping se acumulan como FP densos que gamma=0.7 no olvida.

**Solucion**: Gamma adaptativo por velocidad. Interpola de 0.7 (a < 30 km/h) a 0.0 (a > 72 km/h). Single-frame puro en highway evita acumulacion de errores. F1 highway: 79.7% → 87.6%.

### 5. Paredes Clasificadas como Suelo (Patchwork++)

**Problema**: Patchwork++ ajusta planos por bins CZM. Si un bin contiene la base de una pared, el plano ajustado es vertical y la base se clasifica como ground. Afecta edificios, muros, y la parte baja de vehiculos.

**Solucion**: Rechazo hibrido en dos fases. Fase 1 (bin-wise) detecta bins sospechosos por normal vertical insuficiente. Fase 2 (point-wise) refina con KDTree local para no rechazar bins completos. Rescata ~847 puntos/frame.


## Comparacion con Papers Base

| Metodo | Tipo | F1 | GPU | Entrenamiento | Limitacion principal |
|--------|------|------|-----|---------------|----------------------|
| Cylinder3D | CNN | ~97% | Si | Si | Segmentacion semantica completa (19 clases), problema diferente |
| RangeNet++ | CNN | ~84% | Si | Si | Solo range image, pierde resolucion 3D |
| Dewan et al. | Bayesian | ~83% | No | No | Range image (compresion 20:1), sin gamma, sin egomotion |
| OccAM | Shadow | ~85% | No | Parcial | Solo validacion por sombra, sin temporal |
| **Este trabajo** | **Geometria** | **88.5%** | **No** | **No** | Latencia alta (~5.7s/frame) |

**Mejoras respecto a Dewan et al.**: Per-point 3D (sin range image), gamma adaptativo por velocidad, depth jump check, egomotion compensation, HCD. Resultado: +5.5% F1.

**Mejoras respecto a OccAM**: Filtro temporal Bayesiano previo a validacion por sombra, longitud de sombra geometrica finita basada en altura del obstaculo (no infinita), decaimiento por distancia. Resultado: +3.5% F1.

---

## Estructura del Proyecto

```
sota_idea/
├── lidar_pipeline_suite.py        # Pipeline principal (Stages 1-5, libreria)
├── run_pipeline_viz.py            # Visualizacion por stages en RViz
├── range_projection.py            # Pipeline como nodo ROS 2
├── stage1_visualizer.py           # Visualizacion Stage 1 aislado
│
├── tests/
│   ├── test_full_ablation.py      # Ablation acumulativo + HCD
│   ├── test_adaptive_gamma.py     # Gamma fijo vs adaptativo
│   ├── test_stage4_shadow_validation.py
│   ├── test_stage5_cluster_filtering.py
│   ├── test_full_pipeline_both_sequences.py
│   ├── test_patchwork_wall_problem.py
│   └── archive/                   # Tests obsoletos
│
├── data_kitti/
│   ├── 00/ y 00_labels/           # KITTI seq 00 (urbano)
│   └── 04/ y 04_labels/           # KITTI seq 04 (highway)
│
├── Paso_1/                        # Exploracion inicial de Patchwork++
│   ├── run_patchwork_viz.sh
│   └── visualize_patchwork_rviz.py
│
└── papers/                        # PDFs de papers base
```

---

## Papers Implementados

1. **Patchwork++** (Lee et al., RA-L/IROS 2022) — Stage 1: Ground segmentation con CZM
2. **Dewan et al.** (IROS 2018) — Stage 3: Bayesian temporal filter en log-odds (Eq. 9), adaptado a per-point con gamma adaptativo
3. **ERASOR++** (Lim et al., ICRA 2023) — Stage 1: Height Coding Descriptor
4. **OccAM** (Schinagl et al., CVPR 2022) — Stage 4: Validacion por oclusion, adaptado a KDTree angular con sombra geometrica finita
