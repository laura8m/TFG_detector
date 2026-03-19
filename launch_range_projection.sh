#!/bin/bash
# Script para lanzar range_projection.py con entorno ROS 2 Jazzy configurado

# Configurar ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Agregar pypatchworkpp al PYTHONPATH
export PYTHONPATH="/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/src/patchwork-plusplus/python/pwenv/lib/python3.12/site-packages:$PYTHONPATH"

# Usar Python 3.12 del sistema (compatible con ROS 2 Jazzy)
PYTHON_BIN="/usr/bin/python3.12"

# Script a ejecutar
SCRIPT="/home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/range_projection.py"

# Parámetros por defecto (pueden sobrescribirse con argumentos)
SCAN_START="${1:-0}"
SCAN_END="${2:-0}"

echo "=========================================="
echo "Launching range_projection.py"
echo "=========================================="
echo "Python: $PYTHON_BIN"
echo "ROS_DISTRO: $ROS_DISTRO"
echo "Scan range: $SCAN_START to $SCAN_END"
echo "=========================================="
echo ""

# Ejecutar script
$PYTHON_BIN $SCRIPT \
    --data_path /home/lau8m/lidar_ws/TFG-LiDAR-Geometry/sota_idea/data_kitti/04/04 \
    --scene 04 \
    --scan_start $SCAN_START \
    --scan_end $SCAN_END
