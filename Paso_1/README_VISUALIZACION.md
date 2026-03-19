# 🎯 Visualización de Patchwork++ en RViz

## 🚀 Inicio rápido (1 comando)

```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
./run_patchwork_viz.sh
```

**¡Eso es todo!** El script automáticamente:
- ✅ Ejecuta Patchwork++ vanilla
- ✅ Publica las nubes clasificadas
- ✅ Abre RViz configurado

---

## 📊 Qué verás

### Código de colores en RViz:

```
🟢 VERDE   = Suelo (clasificación correcta en mayoría de casos)
🔴 ROJO    = No-suelo (obstáculos detectados correctamente)
🔵 AZUL    = ⚠️  PROBLEMA: Puntos elevados (Z>0) clasificados como suelo
🟣 MAGENTA = ⚠️  PROBLEMA: Segmentos verticales (paredes) clasificados como suelo
```

### Estadísticas:
- **Total**: 124,668 puntos
- **Suelo**: 72,599 (58.2%)
- **No-suelo**: 52,069 (41.8%)
- **Sospechosos (azul)**: 57 puntos ⚠️
- **Paredes (magenta)**: 389 puntos en 11 segmentos ⚠️

---

## 🔍 Dónde buscar el problema

### Ejemplo de pared mal clasificada:

**Segmento #1** (el más evidente):
- **Ubicación**: X ≈ 26.6m, Y ≈ 4.7m (frontal derecha)
- **Altura**: De -11.56m a -1.71m → **9.85 metros de altura**
- **Color**: MAGENTA
- **Problema**: ¡9.85m de variación vertical! Claramente una PARED, no suelo

### En RViz:
1. Gira la vista para mirar desde arriba (vista XY)
2. Los puntos **MAGENTA** forman líneas/grupos → son las paredes
3. Los puntos **AZUL** están flotando (Z > 0) → también paredes

---

## 📂 Archivos creados

```
sota_idea/
├── visualize_patchwork_rviz.py   # Nodo ROS 2
├── patchwork_debug.rviz           # Configuración RViz
├── run_patchwork_viz.sh           # Script launcher
├── COMO_VISUALIZAR.md             # Guía detallada
└── EVIDENCIA_PATCHWORK.md         # Análisis técnico completo
```

---

## 🎓 Conclusión

Esta visualización **demuestra que Patchwork++ usado tal como indica el README oficial**
clasifica incorrectamente trozos de paredes como suelo.

**Próximo paso**: Implementar el post-procesamiento de rechazo de paredes
que ya está en `range_projection.py` (líneas 394-427).

---

## 📚 Más información

- **Guía detallada**: Ver `COMO_VISUALIZAR.md`
- **Análisis técnico**: Ver `EVIDENCIA_PATCHWORK.md`
- **Scripts de test**: `test_vanilla_patchwork.py`, `debug_patchwork.py`
