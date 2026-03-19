#!/bin/bash
# Script para visualizar Patchwork++ en RViz
# Muestra el problema: paredes clasificadas como suelo

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Visualización de Patchwork++ Vanilla en RViz"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Este script va a:"
echo "  1. Ejecutar Patchwork++ con configuración vanilla (README oficial)"
echo "  2. Publicar las nubes de puntos clasificadas en ROS 2"
echo "  3. Abrir RViz con visualización pre-configurada"
echo ""
echo "Colores en RViz:"
echo "  🟢 VERDE   = Puntos clasificados como SUELO"
echo "  🔴 ROJO    = Puntos clasificados como NO-SUELO"
echo "  🔵 AZUL    = Puntos SOSPECHOSOS (Z > 0, elevados sobre sensor)"
echo "  🟣 MAGENTA = Segmentos VERTICALES (ΔZ > 0.5m, paredes mal clasificadas)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Verificar que estamos en el directorio correcto
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Source ROS 2
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null

# Ejecutar nodo en background
SCAN=${1:-0}
echo "▶️  Iniciando nodo ROS 2 (scan $SCAN)..."
python3 visualize_patchwork_rviz.py --scan $SCAN &
NODE_PID=$!

# Esperar a que el nodo se inicialice
sleep 3

# Verificar que el nodo está corriendo
if ! ps -p $NODE_PID > /dev/null; then
    echo "❌ Error: El nodo no pudo iniciarse"
    exit 1
fi

echo "✅ Nodo iniciado (PID: $NODE_PID)"
echo ""

# Abrir RViz
echo "▶️  Abriendo RViz..."
sleep 1

rviz2 -d patchwork_debug.rviz &
RVIZ_PID=$!

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Visualización activa"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "RViz está abierto. Busca los puntos AZULES (Z>0) y MAGENTA (paredes verticales)."
echo "Estos son puntos clasificados INCORRECTAMENTE como suelo por Patchwork++."
echo ""
echo "Presiona Ctrl+C para detener la visualización."
echo ""

# Trap para cleanup
cleanup() {
    echo ""
    echo "🛑 Deteniendo visualización..."
    kill $NODE_PID 2>/dev/null
    kill $RVIZ_PID 2>/dev/null
    echo "✅ Limpieza completada"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Esperar
wait $NODE_PID $RVIZ_PID
