# Guia de Limpieza — TFG Deteccion de Bordillos con LiDAR

## Pipeline Final

```
Stage 1: Patchwork++ (defaults C++ oficiales) + Wall Rejection (slope=1.0, dz=0.15, r=0.2)
Stage 1.5: Curb Detection (Feature 1: inter-ring height discontinuity) — activable con flag
Stage 2: Delta-r — DESACTIVADO (confirmado que no aporta ni para obstaculos ni para bordillos)
Stage 3: DBSCAN cluster filtering (eps=1.2, min_samples=12, min_pts=10)
```

Archivo principal: `lidar_pipeline_suite.py`

---

## Resultados Clave

### SemanticKITTI + 3D-Curb (Velodyne HDL-64E, 64 beams)

| Config | F1 | Precision | Recall | CurbRecall |
|---|---|---|---|---|
| PW++ vanilla (paper) | 91.2% | 97.2% | 85.6% | — |
| PW++ + WR optimizado | **93.9%** | 95.6% | 92.3% | 7.7% |
| PW++ + WR + Curb Detection | 93.4% | 94.1% | 92.7% | 9.2% |

### GOOSE (Velodyne Alpha Prime, 128 beams)

| Config | F1 | Precision | Recall | CurbRecall |
|---|---|---|---|---|
| PW++ + WR | **97.8%** | 98.7% | 97.0% | 8.3% |
| PW++ + WR + Curb Detection | 95.5% | 92.5% | 98.8% | **58.3%** |

### Comparativa sensor (resultado principal)

| Dataset | Sensor | Beams | CurbRecall |
|---|---|---|---|
| SemanticKITTI | HDL-64E | 64 | 9.2% |
| GOOSE | Alpha Prime | 128 | **58.3%** |

---

## Tests Ejecutados (en orden cronologico)

### 1. Optimizacion Wall Rejection con labels curb=3
- **Script**: `tests/test_wall_rejection_grid_search.py --stride 5`
- **Que hace**: Grid search de slope/dz_threshold/kdtree_radius con labels mapeados de 3D-Curb
- **Resultado**: slope=1.0, dz=0.15, r=0.2 -> F1=93.94% en val (seq 08)
- **Ejecutar en**: mazinger (todas las secuencias)

### 2. Grid Search Curb Detection (Feature 1)
- **Script**: `tests/test_curb_grid_search.py --stride 5`
- **Que hace**: Optimiza curb_height_min/max, curb_min_consecutive, curb_ring_gap
- **Resultado**: Mejor F1=93.39% (cmin=0.15, cmax=0.20, cons=2, gap=1) pero CurbRecall=9.2%
- **Resultado alternativo**: Mejor CurbRecall=58.4% (cmin=0.05, cmax=0.30, gap=4) pero F1=82.4%
- **Conclusion**: Trade-off inevitable en KITTI por resolucion angular del sensor

### 3. Validacion parametros Patchwork++
- **Script**: Prueba inline (no script separado)
- **Que hace**: Compara params actuales vs paper (RNR, delta)
- **Resultado**: Config actual (RNR=False, delta=-1.1) es mejor que paper (+1.5% F1)
- **Justificacion**: RNR requiere intensidad (+0.2% F1 marginal), delta=-1.1 es default del C++ oficial

### 4. Evaluacion en GOOSE (128 beams)
- **Script**: Prueba inline en mazinger
- **Que hace**: Ejecuta pipeline con config adaptada al Alpha Prime (n_rings=128, fov=+-15)
- **Resultado**: CurbRecall sube de 9.2% a 58.3% — confirma limitacion del sensor
- **Datos**: `~/sara/lidar/goose_dataset/goose_3d_val/` (secuencia neubiberg_sunny, 20 frames)

### 5. Comparacion delta-r vs Feature 1 para bordillos
- **Script**: Prueba inline
- **Que hace**: Evalua delta-r con distintos thresholds para detectar bordillos
- **Resultado**: Delta-r es peor que Feature 1 (F1=64% con CurbR=79% vs F1=82% con CurbR=58%)

### 6. Variantes de Feature 1 probadas
- **V1 basico**: DeltaZ inter-ring simple -> muchos FPs
- **V1 + filtro distancia (<15m)**: reduce FPs pero pierde curb lejanos
- **V1 + filtro contexto (nz)**: no aporta (ground ya tiene nz alto)
- **V2 plano-escalon-plano**: F1=92.7%, mejor precision (94.2%) pero CurbRecall=6.9%
- **V2 + DBSCAN + poly fitting (CurbNet)**: demasiado restrictivo, CurbRecall=4%
- **Conclusion**: Ninguna variante resuelve el trade-off en KITTI 64 beams

---

## Archivos Importantes

| Archivo | Rol |
|---|---|
| `lidar_pipeline_suite.py` | Pipeline completo (Stage 1 + curb + DBSCAN) |
| `data_paths.py` | Rutas centralizadas + labels (OBSTACLE/GROUND/IGNORE con curb=3) |
| `scripts/map_3dcurb_labels.py` | Mapea labels 3D-Curb al point cloud completo de SemanticKITTI |
| `run_pipeline_viz.py` | Visualizacion en RViz (--curb para activar bordillos) |
| `tests/test_curb_grid_search.py` | Grid search optimizado con cache de Stage 1 |
| `tests/test_wall_rejection_grid_search.py` | Grid search paralelo de WR |

---

## Datos Necesarios

### SemanticKITTI (en mazinger)
```
~/laura/sota_ws/TFG_detector/data_odometry_velodyne/    # .bin (point clouds)
~/laura/sota_ws/TFG_detector/data_odometry_labels/      # .label (SemanticKITTI original)
~/laura/sota_ws/TFG_detector/data_curb_mapped_labels/   # .label (con curb=3 mapeado de 3D-Curb)
~/laura/sota_ws/TFG_detector/3d_curb_labels/            # 3D-Curb original (bins recortados)
```

### GOOSE (en mazinger)
```
~/sara/lidar/goose_dataset/goose_3d_val/lidar/val/      # .bin (128 beams)
~/sara/lidar/goose_dataset/goose_3d_val/labels/val/     # .label (64 clases, curb=22)
```

---

## Como Reproducir

### Requisitos
```bash
pip install numpy scipy scikit-learn
# Patchwork++ compilado (pypatchworkpp)
```

### Ejecutar pipeline con bordillos (local)
```bash
python3 run_pipeline_viz.py --seq 00 --scan 0 --curb
```

### Ejecutar grid searches (mazinger)
```bash
cd ~/laura/sota_ws/TFG_detector
python3 tests/test_wall_rejection_grid_search.py --stride 5
python3 tests/test_curb_grid_search.py --stride 5
```

### Evaluar en GOOSE (mazinger)
```bash
# Script inline — ver seccion 4 de tests
```

---

## Narrativa del TFG

1. Se implementa un pipeline geometrico de deteccion de obstaculos basado en Patchwork++ con wall rejection optimizado (F1=93.9% en SemanticKITTI)
2. Se propone un detector de bordillos por discontinuidad inter-ring (Feature 1) evaluado con labels de 3D-Curb
3. Se demuestra que la deteccion de bordillos esta limitada por la resolucion angular del sensor: CurbRecall=9.2% con 64 beams vs 58.3% con 128 beams
4. Se concluye que el metodo geometrico es viable para sensores modernos (128+ beams) sin necesidad de deep learning, con F1 global de 95.5% en GOOSE
