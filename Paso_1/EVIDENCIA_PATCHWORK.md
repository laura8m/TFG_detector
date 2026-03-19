# EVIDENCIA: Patchwork++ NO rechaza paredes automáticamente

## Resumen ejecutivo

**Patchwork++ usado tal como se indica en el README oficial clasifica trozos de paredes como suelo.**

## Prueba realizada

### 1. Configuración
- **Script**: `test_vanilla_patchwork.py`
- **Datos**: `000000.bin` del dataset KITTI (124,668 puntos)
- **Configuración**: Patchwork++ con parámetros por defecto (como en el README)
- **Código fuente**: https://github.com/url-kaist/patchwork-plusplus

### 2. Resultados

#### Clasificación básica:
```
Total de puntos:       124,668
Clasificados como SUELO:   72,599 (58.2%)
Clasificados como NO-SUELO: 52,069 (41.8%)
```

#### Problemas detectados:

**A) Puntos elevados clasificados como suelo:**
```
⚠️  57 puntos con Z > 0.0m fueron clasificados como 'suelo'
   → Estos puntos están SOBRE el sensor → Son paredes/obstáculos, NO suelo
```

**B) Segmentos verticales clasificados como suelo:**
```
⚠️  11 segmentos con variación vertical ΔZ > 0.5m
   → Total: 389 puntos (0.5% del suelo clasificado)
   → Segmentos verticales SON PAREDES, no suelo horizontal
```

### 3. Ejemplos de paredes mal clasificadas

#### Segmento #1 - PARED VERTICAL (ΔZ = 9.85m)
```
Ubicación:  Radio = 27.0m, Ángulo = 10° (X≈26.6m, Y≈4.7m)
Puntos:     9 puntos clasificados como SUELO
Rango Z:    -11.56m a -1.71m
Variación:  9.85m ⚠️  EXTREMADAMENTE VERTICAL
```

Este segmento tiene **9.85 metros de variación vertical** en un espacio de 1m×1m radial.
Claramente es una **pared vertical**, NO suelo.

#### Segmento #2 - PARED VERTICAL (ΔZ = 2.85m)
```
Ubicación:  Radio = 29.0m, Ángulo = 25° (X≈26.3m, Y≈12.3m)
Puntos:     6 puntos clasificados como SUELO
Rango Z:    -1.65m a 1.20m
Variación:  2.85m ⚠️  ALTA VERTICALIDAD
```

## Análisis del código fuente de Patchwork++

### Dónde se usa `uprightness_thr` (0.707)

#### 1. Clasificación de puntos (línea 242-267)
```cpp
bool is_upright = ground_uprightness > params_.uprightness_thr;
...
if (!is_upright) {
    addCloud(cloud_nonground_, regionwise_ground_);
}
```
**Efecto**: Clasifica puntos individuales como suelo/no-suelo dentro de cada bin.
**Limitación**: NO elimina el plano del bin.

#### 2. RVPF - Region-wise Vertical Plane Fitting (línea 496)
```cpp
if (zone_idx == 0 && normal_(2) < params_.uprightness_thr) {
    // Rechazar plano vertical SOLO en zona 0
    non_ground_dst.push_back(point);
}
```
**Efecto**: Rechaza planos verticales SOLO en zona 0 (<9.64m).
**Limitación**: En zonas 1, 2, 3 (>9.64m), NO se rechazan planos verticales.

### Por qué `getCenters()` y `getNormals()` devuelven paredes

Patchwork++ genera un plano por cada bin CZM (zona/anillo/sector) que tenga puntos:

1. **Si el bin contiene una pared** → ajusta un plano a esos puntos (RANSAC)
2. **Devuelve el centroide y la normal** del plano ajustado
3. **NO filtra si la normal es horizontal** (excepto zona 0 con RVPF)

**Resultado**: `getCenters()` y `getNormals()` incluyen planos verticales (paredes).

## Distribución de normales en los 274 planos generados

```
Componente vertical (nz) de las normales:
   0.0-0.3 (muy horizontal):   25 planos (9.1%)   ⚠️  PAREDES
   0.3-0.5:                     6 planos (2.2%)   ⚠️  PAREDES
   0.5-0.7:                     7 planos (2.6%)   ⚠️  PAREDES
   0.7-0.85:                    1 plano  (0.4%)
   0.85-0.95:                   8 planos (2.9%)
   0.95-1.0 (muy vertical):   227 planos (82.8%)  ✅  SUELO

Total de planos sospechosos: 38/274 (13.9%)
```

## Conclusión

### ❌ Patchwork++ por sí solo NO es suficiente

Patchwork++ **clasifica puntos** (suelo vs no-suelo) correctamente en la mayoría de los casos,
pero **NO filtra planos verticales** en las zonas lejanas (>9.64m).

### ✅ Solución requerida: Post-procesamiento

Debes implementar **rechazo de paredes** basado en:

1. **Análisis de normales**: `nz < 0.7` → sospechoso de pared
2. **Validación geométrica** (KDTree): `ΔZ > 0.3m` → confirmar pared
3. **Marcar bins rechazados** como obstáculos forzados

### 📖 Implementación de referencia

La lógica completa está en:
- **Archivo**: `range_projection.py`
- **Líneas**: 394-427 (rechazo de paredes)
- **Líneas**: 592-602 (marcado de bins rechazados)

## Scripts de evidencia

1. **`debug_patchwork.py`**: Analiza las normales de los planos generados
2. **`test_vanilla_patchwork.py`**: Test con configuración vanilla (README oficial)
3. **`visualize_wall_problem.py`**: Muestra los 11 segmentos verticales problemáticos

Ejecutar:
```bash
cd ~/lidar_ws/TFG-LiDAR-Geometry/sota_idea
python3 test_vanilla_patchwork.py
python3 visualize_wall_problem.py
```

## Referencias

- **Patchwork++ GitHub**: https://github.com/url-kaist/patchwork-plusplus
- **Paper**: Lim et al., "Patchwork++: Fast and Robust Ground Segmentation Solving Partial Under-Segmentation Using 3D Point Cloud", IROS 2022
- **Código fuente**: `patchworkpp.cpp` líneas 242, 496
