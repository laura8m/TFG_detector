# Tests - Pipeline LiDAR

Esta carpeta contiene los tests para el sistema de detección de obstáculos LiDAR.

## Archivos de Test

### Tests de Módulos del Pipeline

- **`test_pipeline_modules.py`** - Tests unitarios para los módulos individuales del pipeline (ground estimation, delta-r, Bayesian filter, etc.)

### Tests de Rechazo de Paredes

- **`test_wall_rejection_simple.py`** - Implementación simple del rechazo de paredes
- **`test_wall_rejection_v2.py`** - Versión mejorada (v2) del rechazo de paredes con lógica punto-a-punto
- **`test_wall_sweep.py`** - Test de barrido para evaluar diferentes configuraciones de rechazo de paredes

### Tests de Visualización

- **`test_viz_simple.py`** - Tests simples de visualización para validar outputs del pipeline

## Ejecutar los Tests

### Tests individuales

```bash
# Desde el directorio sota_idea
cd /home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea

# Test de módulos del pipeline
python3 tests/test_pipeline_modules.py

# Test de rechazo de paredes v2
python3 tests/test_wall_rejection_v2.py

# Test de visualización
python3 tests/test_viz_simple.py
```

### Todos los tests

```bash
# Ejecutar todos los tests (si usas pytest)
pytest tests/

# O manualmente uno por uno
for test in tests/test_*.py; do python3 "$test"; done
```

## Dependencias

Los tests requieren las mismas dependencias que el proyecto principal:
- NumPy
- SciPy
- Open3D
- ROS 2 (para algunos tests)
- Patchwork++ (módulo compilado)

## Notas

- Algunos tests requieren datos de KITTI en `test_data/sequences/`
- Los tests de visualización pueden requerir display (X11) para mostrar gráficos
- Para tests headless, usar `DISPLAY=:0` o configurar matplotlib backend a 'Agg'
