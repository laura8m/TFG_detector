# Plan de Comparación — TFG Detección de Obstáculos LiDAR

## Protocolo de Evaluación

- **Dataset**: SemanticKITTI
- **Split**: Train 00-07,09-10 / Val 08 / Test 11-21 (sin labels)
- **Secuencia reportable**: **08 (val)** — nunca usada para tuning
- **Métrica**: F1, IoU, Precision, Recall (binario: obstáculo vs no-obstáculo)
- **Mapping obstáculo**: labels 10,11,13,15,16,18,20,30,31,32,50,51,52,70,71,80,81,99,252-259

---

## FASE 1: Resultados propios (HECHO / EN CURSO)

### 1.1 Grid search de parámetros (Mazinger, 128 cores)
- [x] Grid search delta-r + DBSCAN (2880 combos, todas las secuencias)
- [ ] Grid search wall rejection (100 combos, con parámetros óptimos de 1.1)
- [ ] Recoger parámetros óptimos de ambos grid searches

### 1.2 Ablation study en val (seq 08)
Tabla final con resultados en seq 08 usando parámetros óptimos:

| Config | F1 | IoU | P | R | ms/frame |
|--------|----|----|---|---|----------|
| Patchwork++ vanilla (sin wall rejection) | | | | | |
| + Wall Rejection | | | | | |
| + delta-r (Stage 2) | | | | | |
| + delta-r + DBSCAN (Stage 2+3) | | | | | |

---

## FASE 2: Comparación con métodos geométricos

### 2.1 Patchwork++ vanilla
- Ya lo tienes: `test_stage1_ablation.py` compara PW++ vanilla vs PW++ + wall rejection
- Evaluar en seq 08 con la métrica binaria obstáculo
- **No necesita código extra** — ya implementado

### 2.2 RANSAC baseline
- Usar `Open3D` para segmentar suelo con RANSAC
- Todo lo que no sea suelo = obstáculo
- Script simple:
```python
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
plane, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
obs_mask = np.ones(len(points), dtype=bool)
obs_mask[inliers] = False
```

### 2.3 GroundGrid (SOTA geométrico 2024)
- Repo: https://github.com/dcmlr/groundgrid
- Paper: arXiv 2405.15664
- Clonar, compilar, correr en seq 08
- Convertir su salida ground/non-ground a máscara binaria obstáculo

---

## FASE 3: Comparación con métodos deep learning

### Objetivo
Correr modelos DL pre-entrenados en seq 08, convertir sus predicciones
de 19 clases a binario (obstáculo sí/no), y comparar con misma métrica.

### Requisito: GPU con CUDA en Mazinger o Google Colab

### 3.1 Cylinder3D (SOTA DL)
- Repo: https://github.com/xinge008/Cylinder3D
- Pesos: pre-entrenados en SemanticKITTI (disponibles en el repo)
- Pasos:
  1. Clonar repo
  2. Instalar dependencias (PyTorch, spconv)
  3. Descargar pesos pre-entrenados
  4. Correr inferencia en seq 08:
     ```bash
     python test_pretrain_SemanticKITTI.py --config config/semantickitti.yaml \
         --model_path /path/to/pretrained.pt
     ```
  5. Genera archivos .label con predicciones en carpeta output/
  6. Convertir a binario y evaluar

### 3.2 RangeNet++ (baseline DL clásico)
- Repo: https://github.com/PRBonn/lidar-bonnetal
- Pesos: darknet53 pre-entrenado
- Pasos similares a Cylinder3D

### 3.3 SalsaNext (alternativa)
- Repo: https://github.com/TiagoCortinhal/SalsaNext
- Solo si los otros dos dan problemas de instalación

### 3.4 Script de evaluación unificado
Crear un script que lea las predicciones .label de cualquier método
y calcule métricas binarias:

```python
# eval_predictions.py
# Lee .label de un método DL y calcula F1/IoU binario
for scan_id in seq_08_scan_ids:
    pred_labels = np.fromfile(pred_label_file, dtype=np.uint32) & 0xFFFF
    gt_labels = np.fromfile(gt_label_file, dtype=np.uint32) & 0xFFFF

    pred_obs = np.isin(pred_labels, OBSTACLE_LABELS)
    gt_obs = np.isin(gt_labels, OBSTACLE_LABELS)

    tp += np.sum(pred_obs & gt_obs)
    fp += np.sum(pred_obs & ~gt_obs)
    fn += np.sum(~pred_obs & gt_obs)
```

---

## FASE 4: Tabla final para el TFG

| Método | Tipo | F1 (val) | IoU (val) | P | R | ms/frame | GPU | Entrenamiento |
|--------|------|----------|-----------|---|---|----------|-----|---------------|
| RANSAC + todo=obs | Geom. | | | | | | No | No |
| Patchwork++ vanilla | Geom. | | | | | | No | No |
| GroundGrid | Geom. | | | | | | No | No |
| **Nuestro pipeline** | **Geom.** | | | | | | **No** | **No** |
| RangeNet++ | DL | | | | | | Sí | Sí (19k frames) |
| Cylinder3D | DL | | | | | | Sí | Sí (19k frames) |

### Argumentos clave:
- Métodos DL probablemente ganen en F1/IoU, PERO:
  - Requieren GPU (coste, consumo)
  - Requieren datos de entrenamiento etiquetados (19k frames)
  - No son interpretables (caja negra)
  - Domain shift: si cambias de sensor hay que reentrenar
- Nuestro pipeline:
  - Solo CPU, real-time (~53ms sin DBSCAN)
  - Zero-shot: funciona en cualquier LiDAR sin reentrenar
  - Interpretable: cada decisión es explicable
  - Parámetros optimizados con grid search riguroso

---

## Orden de ejecución

1. **AHORA**: Esperar resultados grid search delta-r + DBSCAN (Mazinger)
2. **DESPUÉS**: Lanzar grid search wall rejection con parámetros óptimos
3. **DESPUÉS**: Crear script `eval_predictions.py` para evaluar cualquier método
4. **DESPUÉS**: Correr RANSAC baseline (5 min, no necesita nada)
5. **DESPUÉS**: Intentar GroundGrid (compilar C++)
6. **DESPUÉS**: Intentar Cylinder3D en Mazinger (si tiene GPU) o Colab
7. **FINAL**: Rellenar tabla, escribir sección de resultados del TFG
