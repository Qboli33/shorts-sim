#!/usr/bin/env bash
# Vérifie l'environnement de rendu ; installe OpenCV si absent (pip peut être bloqué : on le signale).
set -u
python3 -c "import numpy, PIL" || { echo "numpy/Pillow manquants"; exit 1; }
python3 -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless 2>&1 | tail -1
python3 -c "import cv2; print('cv2', cv2.__version__)" || { echo "OpenCV indisponible : rendu impossible"; exit 1; }
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 || { echo "ffmpeg sans libx264"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe manquant"; exit 1; }
echo "CPU: $(nproc)"; echo "Environnement OK"
