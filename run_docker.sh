#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_docker.sh  —  Build & run Particle Simulator on Jetson Nano
#
#  Prerequisites on the HOST (Jetson):
#    1. Docker + nvidia-docker2 installed
#    2. X11 display available (monitor plugged in, or SSH -X from a remote PC)
#    3. Camera connected (USB or CSI via /dev/video0)
#
#  Usage:
#    chmod +x run_docker.sh
#    ./run_docker.sh           # build + run
#    ./run_docker.sh --run     # run only (skip rebuild)
# ─────────────────────────────────────────────────────────────────────────────
set -e

IMAGE_NAME="particle-sim-jetson"
CONTAINER_NAME="particle-sim"

# ── Allow X11 connections from Docker ────────────────────────────────────────
xhost +local:docker 2>/dev/null || true

# ── Build image (skip with --run flag) ───────────────────────────────────────
if [[ "$1" != "--run" ]]; then
    echo "[+] Building Docker image: $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" .
fi

# ── Run container ─────────────────────────────────────────────────────────────
echo "[+] Starting container ..."
docker run --rm -it \
    --name "$CONTAINER_NAME" \
    \
    `# GPU access (nvidia-docker2 required)` \
    --runtime nvidia \
    \
    `# Camera device — adjust /dev/video0 if your camera is on a different node` \
    --device /dev/video0:/dev/video0 \
    \
    `# X11 display forwarding` \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    \
    `# Keep the app's working directory writable for any runtime files` \
    -v "$(pwd)":/app \
    \
    "$IMAGE_NAME"

# Revoke X11 access when done
xhost -local:docker 2>/dev/null || true
