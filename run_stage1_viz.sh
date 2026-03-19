#!/bin/bash
# run_stage1_viz.sh
# Script para lanzar visualización Stage 1 en RViz

echo "========================================="
echo "Stage 1 Visualization Launcher"
echo "========================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "stage1_visualizer.py" ]; then
    echo "ERROR: No se encuentra stage1_visualizer.py"
    echo "Ejecuta este script desde: sota_idea/"
    exit 1
fi

# Parsear argumentos
SCAN=0
MODE="single"
ABLATION=""
LOOP=""
SCAN_RANGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scan)
            SCAN="$2"
            shift 2
            ;;
        --ablation)
            MODE="ablation"
            ABLATION="--ablation"
            shift
            ;;
        --loop)
            LOOP="--loop"
            shift
            ;;
        --scan_range)
            SCAN_RANGE="--scan_range $2 $3"
            shift 3
            ;;
        --help|-h)
            echo "Uso: $0 [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --scan N              Procesar scan N (default: 0)"
            echo "  --ablation            Ejecutar ablation study"
            echo "  --loop                Procesar scans en loop continuo"
            echo "  --scan_range N M      Procesar scans del N al M"
            echo "  --help, -h            Mostrar esta ayuda"
            echo ""
            echo "Ejemplos:"
            echo "  $0 --scan 0                    # Visualizar scan 0"
            echo "  $0 --scan 0 --ablation         # Ablation study en scan 0"
            echo "  $0 --scan_range 0 4 --loop     # Loop scans 0-4"
            exit 0
            ;;
        *)
            echo "Argumento desconocido: $1"
            echo "Usa --help para ver opciones"
            exit 1
            ;;
    esac
done

# Paso 1: Lanzar RViz en background
echo "[1/2] Lanzando RViz..."
rviz2 -d stage1_visualization.rviz &
RVIZ_PID=$!

# Esperar a que RViz arranque
sleep 3

# Paso 2: Lanzar visualizer
echo "[2/2] Ejecutando Stage 1 Pipeline..."
echo ""

if [ "$MODE" = "ablation" ]; then
    echo "Modo: ABLATION STUDY"
    echo "Scan: $SCAN"
    echo ""
    echo "Se publicarán 3 configuraciones:"
    echo "  - /stage1/ablation/baseline (verde=ground, naranja=non-ground)"
    echo "  - /stage1/ablation/wall_rejection (+ azul=walls)"
    echo "  - /stage1/ablation/hcd (con HCD)"
    echo ""
    echo "Activa/desactiva los displays en RViz para comparar."
    echo ""

    python3 stage1_visualizer.py --scan $SCAN $ABLATION $LOOP $SCAN_RANGE
else
    echo "Modo: VISUALIZACIÓN SIMPLE"
    echo "Scan: $SCAN"
    echo ""
    echo "Topics publicados:"
    echo "  - /stage1/ground_points (verde)"
    echo "  - /stage1/nonground_points (rojo)"
    echo "  - /stage1/wall_points (azul)"
    echo "  - /stage1/hcd_colored (colormap z_rel)"
    echo "  - /stage1/ground_planes (flechas normales)"
    echo ""

    python3 stage1_visualizer.py --scan $SCAN --enable_hcd $LOOP $SCAN_RANGE
fi

# Cleanup: matar RViz al salir
echo ""
echo "Cerrando RViz..."
kill $RVIZ_PID 2>/dev/null

echo "Finalizado."
