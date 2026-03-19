#!/bin/bash
# Script para probar detección de paredes en múltiples scans de una secuencia

SEQUENCE=${1:-00}
START_SCAN=${2:-0}
END_SCAN=${3:-9}

echo "========================================="
echo "Testing Wall Detection"
echo "========================================="
echo "Sequence: $SEQUENCE"
echo "Scans: $START_SCAN to $END_SCAN"
echo ""

for i in $(seq $START_SCAN $END_SCAN); do
    echo "--- Scan $i ---"
    python3 debug_wall_detection.py --scan $i --sequence $SEQUENCE | grep -E "(Total de puntos:|Planos locales:|Paredes rechazadas:|Vertical|Muy inclinado)"
    echo ""
done

echo "========================================="
echo "Done!"
echo "========================================="
