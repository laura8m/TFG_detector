# Proyección de Rango con Planos Locales y Filtro Bayesiano

Este proyecto implementa un método avanzado de detección de obstáculos en LiDAR utilizando **Planos Locales de Suelo** (estimados por [Patchwork++](https://github.com/url-kaist/patchwork-plusplus)) y un **Filtro Bayesiano Temporal** reforzado por **Validación Geométrica de Sombras (OccAM)**.

El objetivo es detectar obstáculos con alta precisión sobre terrenos irregulares y eliminar falsos positivos (como polvo, hierba, o ruido) mediante consistencia temporal y validación física.

---

## 1. Pipeline del Algoritmo

El sistema sigue un flujo secuencial estricto diseñado para maximizar la robustez:

### 1️⃣ Estimación de Suelo (Patchwork++)
En lugar de asumir un suelo plano global (RANSAC), usamos **Patchwork++** para dividir el entorno en un **Modelo de Zonas Concéntricas (CZM)**.
- Para cada bin (zona/anillo/sector), se ajusta un **Plano Local** ($n_{local}, d_{local}$).
- Esto permite modelar pendientes, baches y aceras correctamente.

### 2️⃣ Detección de Anomalías ($\Delta r$) - Likelihood
Para cada punto, calculamos dónde *debería* caer si fuesen suelo ($r_{exp}$) y lo comparamos con la medición real ($r_{measured}$):
$$ \Delta r = r_{measured} - r_{exp} $$
- $\Delta r \ll 0$: El punto está más cerca de lo esperado -> **Obstáculo**.
- $\Delta r \approx 0$: Suelo.
- **Salida**: Probabilidad Raw ($P_{raw}$).

### 3️⃣ Filtro Bayesiano Temporal (Memoria)
Acumulamos esta evidencia raw en un **Grid Map Bayesiano** (Log-Odds) que viaja con el vehículo (compensación de Egomotion).
- **Check de Salto de Profundidad**: Si un punto cambia drásticamente de distancia (>0.5m) respecto al frame anterior, **NO** heredamos la creencia antigua para evitar conflictos (paredes que aparecen, objetos que se mueven).
- **Salida Actualizada**: Mapa de Log-Odds Temporal.

### 4️⃣ Validación Geométrica de Sombras (Lógica OccAM) - Tratamiento de Oclusiones
Esta es la innovación clave que permite al sistema distinguir entre objetos sólidos (coches, muros) y falsos positivos "transparentes" (polvo, humo, lluvia).

**Principio Físico**:
Un objeto sólido *debe* ocluir la visión del sensor. Si el láser detecta un "obstáculo" en $r=10m$ pero también detecta suelo en $r=15m$ en la misma dirección, entonces el "obstáculo" es, por definición, permeable (humo) o ruido.

**Algoritmo de Boost**:
Para cada celda del mapa con probabilidad de ser obstáculo:
1.  **Ray Casting Inverso**: Trazamos un rayo desde el sensor a través del obstáculo hasta el límite del rango.
2.  **Análisis de la Zona de Sombra**:
    *   **CASO A (Sombra Pura)**: La zona detrás del objeto está **VACÍA** (sin retornos).
        *   $\rightarrow$ **Confirmación**: El objeto es opaco. **Aumentamos su probabilidad (+LogOdds)**.
    *   **CASO B (Penetración)**: La zona detrás contiene puntos clasificados como **SUELO**.
        *   $\rightarrow$ **Contradicción**: El objeto es transparente o inexistente. **Reducimos drásticamente su probabilidad (-LogOdds)**.

**Resultado**:
Los obstáculos reales se "solidifican" rápidamente en el mapa, mientras que el ruido transitorio (que no proyecta sombra consistente) se filtra automáticamente.


### 5️⃣ Suavizado Espacial (Inter-ring Consistency)
Una vez tenemos el mapa de probabilidad Bayesiana potenciado por sombras, aplicamos un filtro espacial (Morfología 2D).
- Conectamos puntos aislados.
- Rellenamos huecos en objetos sólidos.
- Eliminamos ruido "speckle" (puntos sueltos sin vecinos).
- **Salida Final**: $P_{final}$ (Probabilidad definitiva para este frame).

### 6️⃣ Clustering y Visualización - Nivel Objeto
Solo ahora, con la probabilidad $P_{final}$ limpia y robusta, extraemos los objetos:
- Convertimos el mapa de probabilidad a Nube de Puntos 3D.
- Aplicamos **DBSCAN** para agrupar puntos en objetos individuales.
- **Visualización de Sombras (Geometric Shadows)**:
    - Calculamos el **Convex Hull** (envolvente) de cada objeto clustered.
    - Identificamos los vértices extremos en ángulo (izquierda/derecha).
    - Proyectamos estos vértices radialmente hacia afuera hasta un rango límite (aprox. 20m).
    - Generamos un polígono gris que representa la "zona muerta" teórica detrás del objeto.
    - *Nota Importante*: Esta visualización es una **representación sintética a nivel de objeto** para ayudar al humano. El filtro Bayesiano real (Paso 4) trabaja con sombras reales a **nivel de píxel** sin asumir formas.

---

## 2. Visualización en RViz

El nodo publica varios topics para depurar cada etapa del pipeline:

| Topic | Tipo | Descripción |
| :--- | :--- | :--- |
| **`/bayes_cloud`** | `PointCloud2` | **Salida Final**. Nube coloreada por probabilidad de obstáculo (Azul=Alto, Verde=Bajo). Integra Tiempo + Sombras + Espacio. |
| **`/cluster_points`** | `PointCloud2` | Obstáculos confirmados ya agrupados. Cada "objeto" tiene un color aleatorio distinto. |
| **`/shadow_cloud`** | `PointCloud2` | Puntos azules proyectados sobre el suelo. Muestra qué píxeles recibieron "Shadow Boost" positivo. |
| **`/geometric_shadows`** | `Marker` | Polígonos (líneas grises/transparentes) que visualizan la sombra teórica de cada **cluster**. |
| **`/concave_hull`** | `Marker` | Línea magenta que delimita el espacio libre navegable. |
| **`/detected_walls`** | `PointCloud2` | **Debug**. Puntos rojos que muestran los muros rechazados por la validación geométrica (SOTA v2.0). |
| **`/void_cloud`** | `PointCloud2` | **Debug**. Puntos violetas que indican obstáculos negativos (huecos/vacíos) donde falta suelo esperado. |
| `/delta_r_cloud` | `PointCloud2` | **Raw Likelihood**. Lo que ve el sensor en este instante sin memoria ni filtros. |
| `/bayes_temporal_cloud` | `PointCloud2` | **Solo Temporal**. Bayes + Sombras, pero SIN el suavizado espacial final. Útil para ver el efecto puro de la memoria. |
| `/gt_cloud` | `PointCloud2` | Ground Truth (SemanticKITTI) coloreado por clase real. |

### Leyenda de Colores Ground Truth (SemanticKITTI)
Estos son los colores utilizados para visualizar la "Verdad Terrestre" en `/gt_cloud`:

| Clase | Color | RGB |
| :--- | :--- | :--- |
| **Coches / Vehículos** | 🔵 **Azul** | (0, 0, 255) |
| **Personas / Ciclistas** | 🟡 **Amarillo** | (255, 255, 0) |
| **Edificios** | 🔴 **Rojo** | (255, 0, 0) |
| **Vegetación** | 🟢 **Verde** | (0, 150, 0) |
| **Troncos** | 🟤 **Marrón** | (139, 69, 19) |
| **Carretera / Suelo** | 🟣 **Magenta** | (128, 0, 255) |
| **Acera** | 🌸 **Rosa** | (255, 0, 200) |
| **Parking / Valla** | 🟠 **Naranja** | (255, 120, 0) |


---

## 3. Ejecución

### 3.1 Procesar Secuencia (Batch)
Para validar la consistencia temporal, procesa varios frames consecutivos:
```bash
python3 sota_idea/range_projection.py --scan_start 0 --scan_end 4
```

### 3.2 Depuración Paso a Paso
```bash
python3 sota_idea/range_projection.py --scan 0
```

### 3.3 Visualizar
Abrir RViz con la configuración incluida:
```bash
rviz2 -d sota_idea/range_view.rviz
```

---

## 4. Configuración Clave (`range_projection.py`)

- **`self.shadow_decay_dist` (60.0m)**: Distancia a la cual el boost de sombra decae al 20%.
- **`self.threshold_obs` (-0.3m)**: Umbral base de altura para considerar algo un obstáculo potencial.
- **`self.gamma` (0.6)**: Inercia del filtro Bayesiano. Más alto = más rápido olvida el pasado.

---

## 5. Innovaciones Recientes: Concave Hull y Segmentación Robusta

Para mejorar la calidad de la delimitación del espacio libre y la detección en entornos urbanos complejos, hemos introducido las siguientes mejoras técnicas:

### 5.1 Concave Hull Híbrido (Alpha Shapes + Frontier Sampling)
Hemos reemplazado el método de envolvente simple por un pipeline avanzado que combina velocidad y estética:

1.  **Frontier Sampling (Optimización)**:
    *   En lugar de procesar toda la nube, dividimos el espacio polar en **2048 sectores**.
    *   Seleccionamos solo el punto de **rango máximo** por sector.
    *   Esto reduce la entrada de ~100k a ~2k puntos críticos, permitiendo cálculos en milisegundos.

2.  **Inyección de Anclajes de Cluster**:
    *   Para evitar que el submuestreo elimine objetos finos, inyectamos explícitamente los vértices (Bounding Box) de todos los clusters detectados al conjunto de puntos.
    *   Garantiza que **ningún objeto quede fuera** del perímetro navegable.

3.  **Alpha Shapes (Delaunay)**:
    *   Aplicamos Triangulación de Delaunay sobre el conjunto optimizado.
    *   **Mejora v2.0**: Usamos Alpha Adaptativo (ver Sec. 6.2).
    *   Originalmente usábamos un radio fijo de 7.0 metros.

4.  **Suavizado Chaikin**:
    *   Aplicamos 2 iteraciones del algoritmo de "Corner Cutting" al polígono resultante.
    *   Elimina el aspecto "picudo" y genera curvas suaves y naturales.

### 5.2 Segmentación de Suelo con Rechazo de Paredes (Wall Rejection)
Uno de los problemas clásicos es clasificar muros verticales como suelo. Hemos añadido una validación geométrica en la etapa de Patchwork++:

*   **Análisis de Varianza Vertical**: Antes de aceptar un bin como suelo, verificamos si la **normal del plano** estimado es suficientemente vertical (componente Z > 0.7).
*   **Criterio de Rechazo**: Si la normal es demasiado horizontal (pared), el bin se rechaza como suelo.
*   **Mejora v2.0**: Ahora distinguimos entre Muros y Rampas (ver Sec. 6.1).

### 5.3 Justificación Técnica: Prioridad Geométrica vs. Estadística

Es común preguntar si forzar la clasificación de "pared" sobre la probabilidad estadística es "hacer trampas". La respuesta es **no**, es aplicar **Restricciones Físicas Duras (Hard Constraints)** para garantizar la seguridad:

1.  **Evidencia Dura ("Hard Evidence") vs. Estimación Estadística**:
    *   Un filtro Bayesiano suaviza la incertidumbre. Pero si la geometría física (la normal del plano) confirma una superficie vertical, eso es un hecho físico, no una probabilidad.
    *   Ignorar este hecho para promediarlo con datos pasados introduciría latencia peligrosa (el robot podría chocar contra una pared nueva antes de que el filtro "crea" que es real).

2.  **Prioridad de Seguridad (Safety-Critical Design)**:
    *   En robótica móvil, los errores no son simétricos: confundir una pared con suelo (falso negativo) es catastrófico, mientras que frenar ante una falsa pared (falso positivo) es solo ineficiente.
    *   Al aplicar el veto geométrico ("Wall Rejection"), forzamos un comportamiento **conservador**: ante la mínima evidencia de verticalidad, el sistema asume obstáculo inmediatamente, eliminando el "ghosting" o inercia del filtro.

---

## 6. Mejoras del Estado del Arte (SOTA v2.0)

### 6.1. Rechazo de Paredes Sensible a la Pendiente (Validación Geométrica)
La segmentación de suelo estándar (como Patchwork++) a menudo clasifica erróneamente superficies verticales (paredes, postes) como suelo si la geometría local es perfectamente plana. Nuestro enfoque anterior utilizaba un umbral de normal simple (`nz < 0.7`), lo que rechazaba incorrectamente rampas transitables empinadas.

**Nuevo Algoritmo:**
Implementamos una lógica de **Validación Sensible a la Pendiente** que analiza la consistencia de altura local de cada posible plano de suelo "vertical":

1.  **Verificación de Normal**: Primero, identificamos planos con normales horizontales (`abs(nz) < 0.7`).
2.  **Consulta de Geometría Local (KDTree)**: Usamos un `cKDTree` para consultar los puntos LiDAR reales en un radio de **0.5m** del centroide del plano.
3.  **Delta Z Robusto**: Calculamos la variación de altura usando percentiles para ignorar ruido (outliers):
    $$ \Delta Z = P_{95}(z) - P_{5}(z) $$
4.  **Lógica de Decisión**:
    *   **SI** $\Delta Z > 0.3m$ $\rightarrow$ **PARED** (Rechazar). La superficie tiene un escalón vertical significativo (ej. bordillo alto o edificio).
    *   **SI** $\Delta Z \leq 0.3m$ $\rightarrow$ **RAMPA** (Aceptar). La superficie es empinada pero plana localmente (ej. rampa de garaje).
    *   **Fallback**: Si los datos son escasos, se rechaza si el centroide $Z > -1.0m$ (Heurística de altura absoluta).

### 6.2. Alpha Shapes Adaptativas (Concave Hull Dinámico)
Un radio Alpha fijo es subóptimo para datos LiDAR porque la densidad de puntos disminuye cuadráticamente con la distancia. Un radio pequeño (ej. 2m) funciona bien cerca pero fractura el hull a 50m. Un radio grande (ej. 7m) conecta puntos lejanos pero "infla" el hull cerca (perdiendo detalle).

**Nuevo Algoritmo:**
Implementamos un **Umbral Adaptativo** para la triangulación Alpha Shape. El circunradio máximo permitido para un triángulo es dinámico:

$$ R_{max}(d) = \max(4.0, 0.2 \cdot d_{media}) $$

Donde $d_{media}$ es la distancia promedio de los vértices del triángulo al sensor.
*   **A 10m**: $R_{max} = 4.0m$ $\rightarrow$ Ajuste muy fino, contorno detallado.
*   **A 50m**: $R_{max} = 10.0m$ $\rightarrow$ Ajuste relajado, asegurando continuidad a pesar de datos dispersos.
*   **A 80m**: $R_{max} = 16.0m$ $\rightarrow$ Máxima robustez en el límite del rango.

### 6.3. Detección de Obstáculos Negativos (Ground Drop-out)
La mayoría de los sistemas LiDAR fallan al detectar **agujeros** (alcantarillas abiertas, zanjas) o superficies negras absorbentes (hielo negro), ya que el sensor no devuelve puntos en esas zonas (lectura = 0 o Infinito).

**Nuevo Algoritmo (Proactive Void Detection):**
Hemos implementado una lógica que invierte el problema: en lugar de buscar objetos, buscamos la **ausencia de suelo esperado**.

1.  **Predicción de Suelo**: Para cada dirección $(\theta, \phi)$ del láser, calculamos dónde *debería* golpear el suelo según el plano local estimado ($r_{expected}$).
2.  **Verificación de Sensor**: Si el sensor reporta **VACÍO** (ningún retorno) en una dirección donde esperamos suelo cercano ($r_{expected} < 15m$).
3.  **Clasificación**:
    -   $\rightarrow$ **OBSTÁCULO NEGATIVO CONFIRMADO**.
    -   Se inyecta probabilidad de obstáculo en el mapa de Bayes.
    -   Se visualiza como una nube de puntos **VIOLETA** (`/void_cloud`) reconstruida sintéticamente en la posición teórica del suelo faltante.

---
**Autor**: Antigravity (Google DeepMind)
**Fecha**: 16 de Febrero de 2026
