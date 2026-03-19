# 🔍 Cómo visualizar el problema de Patchwork++ en RViz

## Método 1: Script automático (RECOMENDADO)

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
./run_patchwork_viz.sh
```

Este script:
1. ✅ Ejecuta Patchwork++ vanilla
2. ✅ Publica las nubes de puntos clasificadas
3. ✅ Abre RViz con configuración pre-cargada

**Presiona Ctrl+C para detener.**

---

## Método 2: Manual (2 terminales)

### Terminal 1: Ejecutar nodo ROS 2
```bash<>
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
source /opt/ros/jazzy/setup.bash  # o humble
python3 visualize_patchwork_rviz.py
```

### Terminal 2: Abrir RViz
```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
rviz2 -d patchwork_debug.rviz
```

---

## 📊 Qué verás en RViz

### Código de colores:

| Color | Significado | Cantidad |
|-------|-------------|----------|
| 🟢 **VERDE** | Puntos clasificados como SUELO | ~72,599 |
| 🔴 **ROJO** | Puntos clasificados como NO-SUELO | ~52,069 |
| 🔵 **AZUL** (esferas) | Puntos SOSPECHOSOS (Z > 0) | ~57 |
| 🟣 **MAGENTA** | Segmentos VERTICALES (ΔZ > 0.5m) | ~389 |

### ⚠️ **PROBLEMA VISIBLE**

Los puntos **AZULES** y **MAGENTA** están clasificados como SUELO (verde),
pero deberían ser OBSTÁCULOS:

- **AZUL**: Puntos elevados (Z > 0) → están SOBRE el sensor → SON PAREDES
- **MAGENTA**: Segmentos con alta variación vertical (ΔZ > 0.5m) → SON PAREDES VERTICALES

---

## 🎯 Dónde buscar las paredes mal clasificadas

### Vista recomendada en RViz:
1. Usar vista **Orbit**
2. Ajustar ángulo: **Pitch ≈ 45°**, **Yaw ≈ 45°**
3. Zoom out: **Distance ≈ 50m**

### Ubicaciones de paredes (segmentos magenta):

| Ubicación | Radio | Ángulo | ΔZ |
|-----------|-------|--------|-----|
| Pared #1 | 27m | 10° | 9.85m ⚠️ |
| Pared #2 | 29m | 25° | 2.85m ⚠️ |
| Rampa #3 | 5m | -35° | 0.73m |
| Pared #4 | 24m | -20° | 0.69m |

**Coordenadas aproximadas de Pared #1**:
- X ≈ 26.6m, Y ≈ 4.7m
- Rango Z: -11.56m a -1.71m
- **9.85 metros de altura** → Claramente una PARED, NO suelo

---

## 🔧 Controles útiles en RViz

### Ocultar/mostrar capas:
En el panel izquierdo "Displays", activa/desactiva:
- ☑️ `Ground (Verde)` - Mostrar/ocultar suelo
- ☑️ `Non-Ground (Rojo)` - Mostrar/ocultar no-suelo
- ☑️ `Suspicious Z>0 (Azul)` - **IMPORTANTE**: Puntos elevados
- ☑️ `Vertical Segments (Magenta)` - **IMPORTANTE**: Paredes verticales

### Cambiar tamaño de puntos:
1. Click en la capa (ej: "Suspicious Z>0")
2. Expandir → `Size (m)` o `Size (Pixels)`
3. Ajustar valor (ej: 0.05m para esferas más grandes)

---

## 📸 Capturas de pantalla recomendadas

Para documentar el problema, toma capturas desde:

1. **Vista superior (XY)**: Ver distribución espacial de paredes
2. **Vista lateral (XZ)**: Ver altura de paredes (Z > 0)
3. **Vista 3D**: Ver contexto completo

---

## 🐛 Solución de problemas

### Error: "No module named 'rclpy'"
```bash
source /opt/ros/jazzy/setup.bash
# O si usas Humble:
source /opt/ros/humble/setup.bash
```

### Error: "No module named 'pypatchworkpp'"
```bash
# El script ya incluye el path correcto
# Verifica que el entorno pwenv existe:
ls ~/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/
```

### RViz no muestra puntos
1. Verifica que el nodo está corriendo:
   ```bash
   ros2 topic list | grep patchwork
   ```
2. Verifica que hay mensajes:
   ```bash
   ros2 topic echo /patchwork_ground --once
   ```

---

## 📚 Topics publicados

| Topic | Descripción |
|-------|-------------|
| `/patchwork_raw` | Nube original (124,668 puntos) |
| `/patchwork_ground` | Puntos clasificados como suelo (72,599) |
| `/patchwork_nonground` | Puntos clasificados como no-suelo (52,069) |
| `/patchwork_suspicious_walls` | Puntos elevados Z>0 (57) ⚠️ |
| `/patchwork_vertical_segments` | Segmentos ΔZ>0.5m (389) ⚠️ |

---

## 🎓 Conclusión

Esta visualización demuestra que **Patchwork++ vanilla clasifica incorrectamente**:
- 57 puntos elevados (Z > 0) como suelo
- 389 puntos en 11 segmentos verticales (paredes) como suelo

**Solución**: Implementar post-procesamiento de rechazo de paredes (ver `EVIDENCIA_PATCHWORK.md`).
